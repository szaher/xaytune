from trainlib.recipes.align import align, register_reward, reward_registry, dpo_loss, grpo_loss
import trainlib


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

    def test_align_in_recipe_registry(self):
        from trainlib.recipes import recipe_registry
        assert recipe_registry.has("align")

    def test_top_level_align(self):
        assert callable(trainlib.align)

    def test_top_level_align_is_recipe(self):
        from trainlib.recipes.align.align import align as align_fn
        assert trainlib.align is align_fn
