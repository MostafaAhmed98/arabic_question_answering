import gradio as gr
from pathlib import Path
from transformers import pipeline


base_path = str(Path(__file__).parent)
default_question = "من مقدم برنامج خواطر؟"
default_context = "أحمد مازن أحمد أسعد الشقيري  (19 يوليو 1973م، جدة) إعلامي سعودي من أصول فلسطينية بدأ بتقديم برامج فكرية اجتماعية ومضيف السلسلة التليفزيونية خواطر والمضيف السابق لبرنامج يلا شباب، ألّف برامج تلفازية حول مساعدة الشباب على النضج في أفكارهم والبذل في خدمة إيمانهم وتطوير مهاراتهم واكتشاف معرفتهم بالعالم وبدورهم في جعله مكاناً أفضل.[2] اشتهر الشقيري في السعودية والوطن العربي بعد سلسلة برنامج خواطر التي حققت نجاحاً واسعاً نتيجة بساطة أسلوبها ومعالجتها لقضايا الشباب والأمة والتي كانت دائماً تبدأ بمقولته"

def loading_model_and_prediction(question, context):
    # Replace this with your own checkpoint
    model_checkpoint = base_path + "/checkpoint-5769/"
    question_answerer = pipeline("question-answering", model=model_checkpoint)
    predictions = question_answerer(question=question, context=context)
    formated_preds = predictions['answer']
    return formated_preds

def predict(user_question, user_context):
  model_preds = loading_model_and_prediction(user_question, user_context)
  if len(model_preds) == 0:
     return "No answer Found"
  return model_preds


demo = gr.Interface(fn=predict,
                                              inputs=[gr.Text(value= default_question, placeholder="Arabic Question Text", label="Arabic Question Text"),
                                                      gr.Text(value= default_context, placeholder="Arabic Context Text", label="Arabic Context Text")],
                                              outputs=gr.Text(label="Answer Prediction"), title="Arabic Question Answering", allow_flagging=False
                                              )
demo.launch(share=True)