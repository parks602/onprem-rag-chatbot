from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/Phi-4-mini-instruct"  # 예시
AutoTokenizer.from_pretrained(model_id).save_pretrained("./local_models/Phi-4")
AutoModelForCausalLM.from_pretrained(model_id).save_pretrained("./local_models/Phi-4")
