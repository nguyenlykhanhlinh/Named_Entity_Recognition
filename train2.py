# Use a pipeline as a high-level helper
# Use a pipeline as a high-level helper

from transformers import AutoTokenizer, AutoModelForTokenClassification
# Use a pipeline as a high-level helper
from transformers import pipeline

# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = AutoModelForMaskedLM.from_pretrained("vinai/phobert-base-v2")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Liên quan vụ việc CSGT bị tố đánh dân, trúng một cháu nhỏ đang ngủ, đang lan truyền trên mạng xã hội, Đại tá Nguyễn Văn Tảo, Phó Giám đốc Công an tỉnh Tiền Giang vừa có cuộc họp cùng Chỉ huy Công an huyện Châu Thành và một số đơn vị nghiệp vụ cấp tỉnh để chỉ đạo làm rõ thông tin."

ner_results = nlp(example)
print(ner_results)