from lightning.pytorch.cli import LightningCLI
import pyclassify.model
import pyclassify.module
import pyclassify.datamodule

cli = LightningCLI(subclass_mode_data=True, subclass_mode_model=True)