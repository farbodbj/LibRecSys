class BaseMatrixFactorizationModel():
    def __init__(self, user_count: int, item_count: int, latent_dim: int) -> None:
        self.user_count = user_count
        self.item_count = item_count
        self.latent_dim = latent_dim