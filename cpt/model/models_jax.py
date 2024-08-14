import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
from jax import lax
import optax
from numpyro.distributions import constraints
from numpyro.distributions.discrete import ZeroInflatedProbs
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import is_prng_key, promote_shapes


class ZINB(dist.NegativeBinomial2):
    arg_constraints = {
        "mean": constraints.positive,
        "concentration": constraints.positive,
        "gate": constraints.unit_interval,
    }
    support = constraints.nonnegative_integer

    def __init__(self, mean, concentration, gate, *, validate_args=None):
        self.gate = gate
        super(ZINB, self).__init__(mean, concentration, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        key_bern, key_base = random.split(key)
        shape = sample_shape + self.batch_shape

        samples = super(ZINB, self).sample(key_base, sample_shape=sample_shape)
        mask = random.bernoulli(key_bern, self.gate, shape)
        return jnp.where(mask, 0, samples)

    def log_prob(self, value):
        log_prob = super(ZINB, self).log_prob(value)
        log_prob = jnp.log1p(-self.gate) + log_prob
        return jnp.where(value == 0, jnp.log(self.gate + jnp.exp(log_prob)), log_prob)


class FlaxEncoder(nn.Module):
    """Encoder for Jax VAE."""

    n_input: int
    n_latent: int
    n_hidden: int
    precision: str

    def setup(self):
        """Setup encoder."""
        self.dense1 = nn.Dense(self.n_hidden, precision=self.precision)
        self.dense2 = nn.Dense(self.n_hidden, precision=self.precision)
        self.dense3 = nn.Dense(self.n_latent, precision=self.precision)
        self.dense4 = nn.Dense(self.n_latent, precision=self.precision)

        self.norm1 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        self.norm2 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        # self.norm1 = nn.LayerNorm()
        # self.norm2 = nn.LayerNorm()

    def __call__(self, x: jnp.ndarray, training: bool = False):
        """Forward pass."""
        is_eval = not training

        x_ = jnp.log1p(x)

        h = self.dense1(x_)
        h = self.norm1(h, use_running_average=is_eval)
        # h = self.norm1(h)
        h = nn.relu(h)
        # h = self.dense2(h)
        h = self.norm2(h, use_running_average=is_eval)
        # h = self.norm2(h)
        h = nn.relu(h)

        mean = self.dense3(h)
        log_var = self.dense4(h)
        return dist.Normal(mean, nn.softplus(log_var))


class FlaxDecoder(nn.Module):
    """Decoder for Jax VAE."""

    n_input: int
    n_hidden: int
    precision: str

    def setup(self):
        """Setup decoder."""
        self.dense1 = nn.Dense(self.n_hidden, precision=self.precision)
        self.dense2 = nn.Dense(self.n_hidden, precision=self.precision)
        self.dense3 = nn.Dense(self.n_hidden, precision=self.precision)
        self.dense4 = nn.Dense(self.n_hidden, precision=self.precision)
        self.dense5 = nn.Dense(self.n_input, precision=self.precision)

        self.norm1 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        self.norm2 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        self.disp = self.param(
            "disp", lambda rng, shape: jax.random.normal(rng, shape), (self.n_input, 1)
        )

        self.zi_logits1 = nn.Dense(self.n_hidden)
        self.zi_logits2 = nn.Dense(self.n_hidden)
        self.zi_logits3 = nn.Dense(self.n_input)
        self.zi_logits_norm = nn.BatchNorm(momentum=0.99, epsilon=0.001)

    def __call__(self, z: jnp.ndarray, batch: jnp.ndarray, training: bool = False):
        """Forward pass."""
        is_eval = not training

        h = self.dense1(z)
        h += self.dense2(batch)

        h = self.norm1(h, use_running_average=is_eval)
        # h = self.norm1(h)
        h = nn.relu(h)
        h = self.dense3(h)
        # skip connection
        # h += self.dense4(batch)
        h = self.norm2(h, use_running_average=is_eval)
        # h = self.norm2(h)
        h = nn.relu(h)
        h = self.dense5(h)
        h = nn.softmax(h)

        logits = self.zi_logits1(z)
        logits += self.zi_logits2(batch)
        logits = self.zi_logits_norm(logits, use_running_average=is_eval)
        logits = nn.relu(logits)
        logits = self.zi_logits3(logits)
        probs = nn.sigmoid(logits)
        return h, self.disp.ravel(), probs
    
    
class FlaxLinearDecoder(nn.Module):
    """Decoder for Jax VAE."""

    n_input: int
    n_hidden: int
    precision: str

    def setup(self):
        """Setup decoder."""
        self.dense1 = nn.Dense(self.n_input, precision=self.precision)
        self.dense2 = nn.Dense(self.n_input, precision=self.precision)
        self.disp = self.param(
            "disp", lambda rng, shape: jax.random.normal(rng, shape), (self.n_input, 1)
        )
        self.zi_logits1 = nn.Dense(self.n_hidden)

    def __call__(self, z: jnp.ndarray, batch: jnp.ndarray, training: bool = False):
        """Forward pass."""

        h = self.dense1(z)
        h += self.dense2(batch)
        h = nn.softmax(h)

        logits = self.zi_logits1(z)
        probs = nn.sigmoid(logits)
        return h, self.disp.ravel(), probs


class SCVICRT(nn.Module):
    n_input: int
    n_latent: int
    n_hidden: int
    precision: str
    likelihood: str = "nb"
    dropout_rate: float = 0.0
    linear_decoder: bool = False

    def setup(self):
        self.encoder = FlaxEncoder(
            self.n_input, self.n_latent, self.n_hidden, precision=self.precision
        )
        if self.linear_decoder:
            self.decoder = FlaxLinearDecoder(
                self.n_input, self.n_hidden, precision=self.precision
            )
        else:
            self.decoder = FlaxDecoder(
                self.n_input, self.n_hidden, precision=self.precision
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        x,
        batch_indices,
        n_samples=1,
        training: bool = False,
        use_prior=False,
        kl_weight=1.0,
    ):
        z_rng = self.make_rng("z")
        sample_shape = () if n_samples == 1 else (n_samples,)
        x_ = jnp.log1p(x)
        x_ = self.dropout(x_, deterministic=not training)
        if use_prior:
            qz = dist.Normal(0, 1)
        else:
            qz = self.encoder(x_, training=training)
        z = qz.rsample(z_rng, sample_shape=sample_shape)

        h, disp, probs = self.decoder(z, batch_indices, training=training)
        library = x.sum(-1, keepdims=True)
        if self.likelihood == "nb":
            px = dist.NegativeBinomial2(h * library, jnp.exp(disp))
        elif self.likelihood == "zinb":
            # px = dist.ZeroInflatedNegativeBinomial2(
            #     h * library, jnp.exp(disp), gate_logits=logits
            # )
            px = ZINB(h * library, jnp.exp(disp), probs)
        else:
            px = dist.Poisson(h * library)
        log_px = px.log_prob(x).sum(-1)
        kl = dist.kl_divergence(qz, dist.Normal(0, 1)).sum(-1)
        elbo = log_px - (kl_weight * kl)
        loss = -elbo.mean()
        reconstruction_loss = -log_px.mean()
        return dict(
            loss=loss, h=h, z=z, px=px, reconstruction_loss=reconstruction_loss, qz=qz
        )


class ImportanceScorer(nn.Module):
    n_hidden: int
    n_features: int
    dropout_rate: float
    loss_type: str = "mse"
    residual: bool = False
    activation: str = "relu"

    def setup(self):
        self.dense1 = nn.Dense(features=self.n_hidden)
        self.dense_res = nn.Dense(features=self.n_features)
        self.norm1 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        # self.norm1 = nn.LayerNorm()
        self.dropout1 = nn.Dropout(rate=self.dropout_rate)
        # self.dense2 = nn.Dense(features=self.n_features)

        # self.dense2 = nn.Dense(features=self.n_hidden)
        # self.norm2 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        # self.norm2 = nn.LayerNorm()
        self.dense3 = nn.Dense(features=self.n_features)

        # self.log_std = self.param(
        #     "log_std",
        #     lambda rng, shape: jax.random.normal(rng, shape),
        #     (self.n_features,),
        # )
        self.log_std = 0.0
        if self.activation == "relu":
            self.activation_fn = nn.relu
        elif self.activation == "leaky_relu":
            self.activation_fn = nn.leaky_relu
        elif self.activation == "tanh":
            self.activation_fn = nn.tanh

    def __call__(self, x, y, training: bool = False):
        is_eval = not training

        h = self.dense1(x)
        h = self.norm1(h, use_running_average=is_eval)
        # h = self.norm1(h)
        h = self.activation_fn(h)
        h = self.dropout1(h, deterministic=is_eval)
        if self.residual:
            h = self.dense3(h) + self.dense_res(x)
        else:
            h = self.dense3(h)
        # h = self.dense2(h)
        # h = self.dense_res(x) + h
        # all_loss = -dist.Normal(h, jnp.exp(self.log_std)).log_prob(y)

        # h = self.dense2(h)
        # h = self.norm2(h, use_running_average=is_eval)
        # # h = self.norm2(h)
        # h = nn.leaky_relu(h)
        # h = self.dense3(h) + self.dense_res(x)


        if self.loss_type == "mse":
            all_loss = -dist.Normal(h, jnp.exp(self.log_std)).log_prob(y)
        elif self.loss_type == "binary":
            all_loss = -dist.Bernoulli(logits=h).log_prob(y)
        elif self.loss_type == "huber":
            all_loss = optax.huber_loss(h - y)
        elif self.loss_type == "poisson":
            rate = nn.softplus(h)
            all_loss = -dist.Poisson(1e-6 + rate).log_prob(y)
        loss = all_loss.mean()
        return dict(h=h, loss=loss, all_loss=all_loss)


class ImportanceScorerLinear(nn.Module):
    n_features: int
    loss_type: str = "mse"

    def setup(self):
        self.dense1 = nn.Dense(features=self.n_features)
        self.norm1 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        self.dropout1 = nn.Dropout(rate=0.0)
        # self.log_std = self.param(
        #     "log_std",
        #     lambda rng, shape: jax.random.normal(rng, shape),
        #     (self.n_features,),
        # )
        self.log_std = 0.0

    def __call__(self, x, y, training: bool = False):
        is_eval = not training
        h = self.norm1(x, use_running_average=is_eval)
        h = self.dropout1(h, deterministic=is_eval)
        h = self.dense1(h)
        # h = self.dense1(x)
        if self.loss_type == "mse":
            all_loss = -dist.Normal(h, jnp.exp(self.log_std)).log_prob(y)
        elif self.loss_type == "binary":
            all_loss = -dist.Bernoulli(logits=h).log_prob(y)
        loss = all_loss.mean()
        return dict(h=h, loss=loss, all_loss=all_loss)


class GibbsEncoder(nn.Module):
    n_input: int
    n_latent: int
    n_hidden: int
    precision: str

    def setup(self):
        self.amats = nn.Embed(
            self.n_input, self.n_input * self.n_hidden, dtype=self.precision
        )
        self.bvecs = nn.Embed(self.n_input, self.n_hidden, dtype=self.precision)

        self.dense1 = nn.Dense(self.n_hidden, precision=self.precision)
        self.dense3 = nn.Dense(self.n_latent, precision=self.precision)
        self.dense4 = nn.Dense(self.n_latent, precision=self.precision)

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

    def __call__(self, x, masked_genes, training: bool = False):
        masked_genes = masked_genes.astype(jnp.int32)
        x_ = x.at[:, masked_genes].set(0)
        masks_oh = jax.nn.one_hot(masked_genes, self.n_input)
        x_ = (1.0 - masks_oh) * x_
        x_ = jnp.log1p(x_)
        amats = self.amats(masked_genes)
        bvecs = self.bvecs(masked_genes)
        amats = amats.reshape((x.shape[0], self.n_hidden, self.n_input))
        h = jnp.einsum("ijk,ik->ij", amats, x_)
        h += bvecs
        # h = x_ @ self.amats[masked_gene] + self.bvecs[masked_gene]
        h = self.dense1(h)
        h = self.norm1(h)
        h = nn.relu(h)

        mean = self.dense3(h)
        log_var = self.dense4(h)
        return dist.Normal(mean, jnp.exp(log_var))


class GibbsSCVI(nn.Module):
    n_input: int
    n_latent: int
    n_hidden: int
    precision: str

    def setup(self):
        self.encoder = GibbsEncoder(
            self.n_input, self.n_latent, self.n_hidden, self.precision
        )
        self.decoder = FlaxDecoder(self.n_input, self.n_hidden, self.precision)

    def __call__(
        self, x, batch_indices, masked_genes, n_samples=1, training: bool = False
    ):
        z_rng = self.make_rng("z")
        sample_shape = () if n_samples == 1 else (n_samples,)

        qz = self.encoder(x, masked_genes, training=training)
        z = qz.rsample(z_rng, sample_shape=sample_shape)

        h, disp = self.decoder(z, batch_indices, training=training)
        l = x.sum(-1, keepdims=True)
        # px = dist.NegativeBinomial2(h * l, jnp.exp(disp))
        px = dist.Poisson(h * l)
        log_px = px.log_prob(x)
        kl = dist.kl_divergence(qz, dist.Normal(0, 1)).sum(-1)
        elbo = log_px.sum(-1) - kl
        loss = -elbo.mean()
        return dict(loss=loss, h=h, z=z, px=px, all_loss=-log_px)

    def decode_z(self, x, z, batch_indices, training: bool = False):
        h, disp = self.decoder(z, batch_indices, training=training)
        library = x.sum(-1, keepdims=True)
        # px = dist.NegativeBinomial2(h * library, jnp.exp(disp))
        px = dist.Poisson(h * library)
        return px

    def decode_z_onegene(self, x, z, gene_idx, batch_indices, training: bool = False):
        h, _ = self.decoder(z, batch_indices, training=training)
        library = x.sum(-1, keepdims=False)
        px = dist.Poisson(h[:, gene_idx] * library)
        return px
