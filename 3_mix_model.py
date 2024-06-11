# 融合模型
from LM_Cocktail import mix_models

# Mix fine-tuned model and base model; then save it to output_path: ./mixed_model_1
model = mix_models(
    model_names_or_paths=["model/bge-m3", "model/trianed_bgem3"],
    model_type="encoder",
    weights=[0.5, 0.5],  # you can change the weights to get a better trade-off.
    output_path="/data/zz/kdd_race2/model/mixed_model_5_5",
)
