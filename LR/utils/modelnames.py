from models.mf import mf
from models.ips_mf import ipsmf
from models.featuree_mf import featureemf
from models.delay_mf import delaymf
from models.weightc_mf import weightcmf
from models.cause_mf import causemf
from models.bridge_var1_mf import bridgemf1
from models.bridge_var2_mf import bridgemf2
from models.refine_mf import refinemf
from models.old_bridge_mf import oldbridgemf
from models.old_refine_mf import oldrefinemf
from models.old_weightc_mf import oldweightcmf

models = {
    "MF": mf,
    "IPS-MF": ipsmf,
    "FeatureE-MF": featureemf,
    "Delay-MF": delaymf,
    "WeightC-MF": weightcmf,
    "CausE-MF": causemf,
    "Bridge-Var1-MF": bridgemf1,
    "Bridge-Var2-MF": bridgemf2,
    "Refine-MF": refinemf,
    "Old-Bridge-MF": oldbridgemf,
    "Old-Refine-MF": oldrefinemf,
    "Old-WeightC-MF": oldweightcmf,
}

