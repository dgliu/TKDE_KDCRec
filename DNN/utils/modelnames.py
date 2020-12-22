from models.ae import ae
from models.ips_ae import ipsae
from models.bridge_var1_ae import bridgeae1
from models.bridge_var2_ae import bridgeae2
from models.refine_ae import refineae
from models.weightc_ae import weightcae
from models.delay_ae import delayae
from models.featuree_ae import featureeae
from models.old_bridge_ae import oldbridgeae
from models.old_refine_ae import oldrefineae
from models.old_weightc_ae import oldweightcae

models = {
    "AE": ae,
    "IPS-AE": ipsae,
    "Bridge-Var1-AE": bridgeae1,
    "Bridge-Var2-AE": bridgeae2,
    "Refine-AE": refineae,
    "WeightC-AE": weightcae,
    "Delay-AE": delayae,
    "FeatureE-AE": featureeae,
    "Old-Bridge-AE": oldbridgeae,
    "Old-Refine-AE": oldrefineae,
    "Old-WeightC-AE": oldweightcae,
}

