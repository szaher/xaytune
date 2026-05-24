import xaytune
from xaytune import export


class TestTopLevelAPI:
    def test_version(self):
        assert xaytune.__version__ == "0.6.0"

    def test_finetune_importable(self):
        assert callable(xaytune.finetune)

    def test_pretrain_importable(self):
        assert callable(xaytune.pretrain)

    def test_align_importable(self):
        assert callable(xaytune.align)

    def test_evaluate_importable(self):
        assert callable(xaytune.evaluate)

    def test_export_module(self):
        assert callable(export.merge)
        assert callable(export.save)
        assert callable(export.push_to_hub)

    def test_finetune_is_recipe(self):
        from xaytune.recipes.finetune import finetune

        assert xaytune.finetune is finetune

    def test_pretrain_is_recipe(self):
        from xaytune.recipes.pretrain import pretrain

        assert xaytune.pretrain is pretrain

    def test_align_is_recipe(self):
        from xaytune.recipes.align.align import align

        assert xaytune.align is align

    def test_evaluate_is_eval(self):
        from xaytune.eval.evaluate import evaluate

        assert xaytune.evaluate is evaluate
