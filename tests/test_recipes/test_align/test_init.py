import trainlib
from trainlib.recipes.align import (
    align,
    dpo_loss,
    grpo_loss,
    orpo_loss,
    ppo_clip_loss,
    ppo_value_loss,
    register_reward,
    reinforce_loss,
    reward_registry,
    simpo_loss,
)


class TestAlignPublicAPI:
    def test_align_importable(self):
        assert callable(align)

    def test_register_reward_importable(self):
        assert callable(register_reward)

    def test_reward_registry_importable(self):
        assert reward_registry is not None

    def test_dpo_loss_importable(self):
        assert callable(dpo_loss)

    def test_grpo_loss_importable(self):
        assert callable(grpo_loss)

    def test_orpo_loss_importable(self):
        assert callable(orpo_loss)

    def test_simpo_loss_importable(self):
        assert callable(simpo_loss)

    def test_ppo_clip_loss_importable(self):
        assert callable(ppo_clip_loss)

    def test_ppo_value_loss_importable(self):
        assert callable(ppo_value_loss)

    def test_reinforce_loss_importable(self):
        assert callable(reinforce_loss)

    def test_align_in_recipe_registry(self):
        from trainlib.recipes import recipe_registry

        assert recipe_registry.has("align")

    def test_top_level_align(self):
        assert callable(trainlib.align)

    def test_top_level_align_is_recipe(self):
        from trainlib.recipes.align.align import align as align_fn

        assert trainlib.align is align_fn
