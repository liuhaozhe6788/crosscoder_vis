from sae_vis.data_config_classes import SaeVisConfig
test_feature_idx = [2325,12698,15]
sae_vis_config = SaeVisConfig(
    hook_point = folded_cross_coder.cfg["hook_point"],
    features = test_feature_idx,
    verbose = True,
    minibatch_size_tokens=4,
    minibatch_size_features=16,
)

from sae_vis.data_storing_fns import SaeVisData
sae_vis_data = SaeVisData.create(
    encoder = sae_vis_cross_coder,
    encoder_B = None,
    model_A = base_model,
    model_B = chat_model,
    tokens = all_tokens[:128], # in practice, better to use more data
    cfg = sae_vis_config,
)

filename = "_feature_vis_demo.html"
sae_vis_data.save_feature_centric_vis(filename)