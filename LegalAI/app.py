from flask import Flask,render_template,request
import torch
import pickle
import os
from langdetect import detect
import lawgpt.app as l
from translate import Translator
from lawgpt.utils import qa_pipeline
from transformers import BertForSequenceClassification
import google.generativeai as genai
from chatbot import scanned_image_to_text,chatting
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
model.load_state_dict(torch.load('model.pth',map_location=torch.device('cpu')))

def translate(text, src, dest):
    translator = Translator(to_lang=dest, from_lang=src)
    result = ""
    text_chunks = [text[i:i+500] for i in range(0, len(text), 500)]  # Split text into chunks of maximum 500 characters
    for chunk in text_chunks:
        result += translator.translate(chunk)
    return result

file = 'uploads/hello.pdf'

google_api_key=os.environ.get('GOOGLE_API_KEY')
google_api_key = 'AIzaSyCUztV7AX_8kV2tBCxDetvXsa1HKwz9p1A'
genai.configure(api_key=google_api_key)
model1 = genai.GenerativeModel("gemini-pro")


app=Flask(__name__)
UPLOAD_FOLDER = 'uploads'

# Check if the upload directory exists, if not, create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set the upload folder in the Flask app configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")

def index():
    return render_template("newhome.html")

@app.route("/page1")
def page1():
    return render_template("index.html",title='Page1')

@app.route('/page3')
def page3():
    return render_template('index3.html', title='Page3')

@app.route('/page2')
def page2():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PDF File Upload</title>
        <style>
            body {
                background-color: rgb(44, 47, 59); /* Grey background */
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh; /* Full height of viewport */
                margin: 0;
            }
            .container {
                height: 500px;
                background-color: rgba(0,0,0,0.4); /* Darker grey container */
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }
            .logo-container {
                display: flex;
                justify-content: center;
                align-items: center;
                margin-bottom: 20px;
            }
            .logo {
                width: 100px; /* Adjust the size of the logo */
                height: 100px;
                border-radius: 50%; /* Make the logo rounded */
                margin-right: 10px; /* Add margin between the logo and text */
            }
            .chatbot-info {
                text-align: left;
                color: #fff; /* Text color */
                font-size: 24px; /* Bigger font size for the "ChatBot" text */
            }
            input[type="file"] {
                display: block;
                margin: 20px auto; /* Center the attachment input */
                padding: 10px;
                border: 2px solid #fff; /* White border */
                border-radius: 5px; /* Rounded corners */
                background-color: transparent; /* Transparent background */
                color: #fff; /* White text color */
            }
            button {
                display: block;
                margin: 0 auto; /* Center the upload button */
                padding: 10px 20px; /* Add padding */
                border: none; /* Remove default button border */
                border-radius: 5px; /* Rounded corners */
                background-color: #007bff; /* Blue button color */
                color: #fff; /* White text color */
                cursor: pointer; /* Change cursor to pointer */
            }
            button:hover {
                background-color: #0056b3; /* Darker blue color on hover */
            }
        </style>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    </head>
    <body>
        <div class="container">
            <div class="logo-container">
                <img src="https://i.ibb.co/fSNP7Rz/icons8-chatgpt-512.png" class="logo">
                <div class="chatbot-info">
                    <span style="font-weight: bold; font-size: 32px;">ChatBot</span>
                </div>
            </div>
            <h2 style="color: #F5F5F5;">Upload PDF</h2>
            <input type="file" id="pdf-file" accept=".pdf" required style="width: 200px;">
            <button id="upload-btn">Upload</button>
        </div>

        <script>
            $(document).ready(function() {
                $('#upload-btn').click(function() {
                    var fileInput = document.getElementById('pdf-file');
                    var file = fileInput.files[0];
                    var formData = new FormData();
                    formData.append('pdf_file', file);

                    $.ajax({
                        url: '/upload',
                        type: 'POST',
                        data: formData,
                        contentType: false,
                        processData: false,
                        success: function(response) {
                            alert(response);
                            window.location.href = '/page4';
                        },
                        error: function(xhr, status, error) {
                            console.error(xhr.responseText);
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    '''



@app.route('/page4')
def page4():
    return render_template('index2.html')
@app.route('/upload', methods=['GET','POST'])
def upload():
    if 'pdf_file' not in request.files:
        return 'No file part'
    file = request.files['pdf_file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = 'hello.pdf'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return 'Our commitment to GDPR compliance ensures your data is handled with the highest standards of protection and respect for your privacy.'

@app.route("/get",methods=["GET","POST"])

def chat():
    msg=request.form["msg"]
    inp=msg
    return get_resp(inp)

def get_resp(inp):
    #for step in range(5):
    inputs = tokenizer(str(inp), padding=True, truncation=True, max_length=100, return_tensors="pt")

# Get the model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

# Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=1)[0]

# Print the prediction scores
    print("Prediction scores:", probs)
    # Dictionary mapping section numbers to penalty and additional information
    section_info = {
        'Section 326 in The Indian Penal Code': {
            'Penalty': 'Conviction under Section 326 can lead to imprisonment for up to ten years, or life imprisonment, along with a fine. The exact punishment depends on the severity of the injury and the circumstances of the case.',
            'Additional Information': 'Section 326 A: This section specifically deals with throwing or attempting to throw acid with the intention of causing permanent or partial damage to a person. Section 326 B: This section criminalizes the act of voluntarily causing grievous hurt by using any explosive substance.'
        },
        'Section 354 in The Indian Penal Code': {
            'Penalty': 'The punishment under Section 354 depends on the severity of the offense and has been enhanced in recent years: Imprisonment: For a term of one to five years, with the possibility of either simple or rigorous imprisonment. Fine: The court can also impose a fine. Enhanced Penalty (2013 Amendment): In certain cases, the minimum punishment is now one year, with a maximum of seven years. These cases include situations where the offender is in a position of trust or authority towards the woman (e.g., teacher, employer) or if the act involves disrobing the woman.',
            'Additional Information': 'Related Sections: The IPC has other sections that address various forms of harassment and violation of a woman\'s modesty: Section 354A: Sexual harassment Section 354B: Disrobing a woman Section 354D: Stalking Reporting a Crime: If you have been a victim of a crime under Section 354 IPC, you can report it to the police. There are also helplines and NGOs that can provide support and guidance.'
        },
        'Section 375 in The Indian Penal Code': {
            'Penalty': 'The punishment for rape under Section 375 is severe and can vary depending on the circumstances: Minimum Imprisonment: Not less than seven years, and can extend to life imprisonment. Fine: The court can also impose a fine.',
            'Additional Information': 'Gender Neutrality Debate: Section 375 is currently being debated for its gender bias as it only recognizes men as perpetrators and women as victims. Legal Amendments: The Criminal Law Amendment Act of 2013 has expanded the definition of rape to include new categories like penetration by objects and sexual assault on a transgender person.'
        },
        'Section 376 in The Indian Penal Code': {
            'Penalty': 'Section 376 deals specifically with the punishment for those convicted of rape as defined under Section 375 of the IPC. Recent Amendments (2013): The Criminal Law Amendment Act of 2013 significantly strengthened the punishment structure for rape: Minimum Sentence: Introduced a minimum sentence of 20 years for rape involving a minor below 12 years old. Enhanced Penalties: Increased penalties for gang rape and rape committed by someone in a position of trust or authority towards the victim. Gender Neutrality Considerations: Similar to Section 375, discussions are ongoing regarding incorporating gender neutrality into Section 376 to recognize victims beyond women.',
            'Additional Information': 'Recent Amendments (2013): The Criminal Law Amendment Act of 2013 significantly strengthened the punishment structure for rap Minimum Sentence: Introduced a minimum sentence of 20 years for rape involving a minor below 12 years old Enhanced Penalties: Increased penalties for gang rape and rape committed by someone in a position of trust or authority towards the victim. Gender Neutrality Considerations: Similar to Section 375, discussions are ongoing regarding incorporating gender neutrality into Section 376 to recognize victims beyond women. add to section 376'
        },
        'Section 509 in The Indian Penal Code': {
            'Penalty': 'The offense under Section 509 is considered a cognizable offense, meaning that police can arrest the accused without a warrant. It is also a bailable offense, allowing the accused to apply for bail. The offense is compoundable, meaning it can be settled between the accused and the victim with the court\'s permission. Recent developments have seen courts interpret Section 509 to encompass a wider range of actions, including sending offensive messages or emails. Section 509 is distinct from sexual harassment laws, as it focuses on one-time incidents, whereas laws like the Protection of Women at Workplace Act (POSHA) deal with repeated behavior that creates a hostile work environment.',
            'Additional Information': 'Recent Developments: Courts have interpreted Section 509 to encompass a wider range of actions, including sending offensive messages or emails.Distinction from Sexual Harassment: While Section 509 focuses on one-time incidents, the Protection of Women at Workplace Act (POSHA) deals with repeated behavior that creates a hostile work environment'
        }
    }


    

    idx2label={0: 'Section 326 in The Indian Penal Code', 1: 'Section 354 in The Indian Penal Code', 2: 'Section 375 in The Indian Penal Code', 3: 'Section 376 in The Indian Penal Code', 4: 'Section 509 in The Indian Penal Code'}
    # Get the predicted label indices and corresponding probabilities
    predicted_label_indices = torch.argsort(probs, descending=True)
    predicted_labels = []
    prev_prob = None
    threshold = 0.1
    # Iterate over the probabilities and select labels based on the difference threshold
    for idx in predicted_label_indices:
        prob = probs[idx].item()
        if prev_prob is not None and prev_prob - prob > threshold:
            break
        predicted_labels.append(idx2label[idx.item()])
        prev_prob = prob

    print("Predicted Labels:", predicted_labels)
    outer_response = 'The case mentioned falls under:' + '<br>'
    # Iterate over the predicted labels and construct the response
    response = '' 
    for label in predicted_labels:
        response = ''
        response += label + '<br>' + ' Penalty:<br>' + section_info[label]['Penalty'] + '<br>'
        outer_response = outer_response + '\n\n' + response 
    #response='The case mentioned falls under :'+'<br>'
    #for i in predicted_labels:
    #    response = response + str(i) + '<br>'    

# Print the predicted label and its corresponding penal code
    return outer_response
@app.route('/get1', methods=['GET','POST'])
def get1():
    msg=request.form["msg"]
    inp=msg
    lang = detect(str(inp))
    return get_resp_1(str(inp), lang)
    
def get_resp_1(inp, lang):
    if lang == 'ta':
        inp = translate(KeyboardInterrupt, 'ta' , 'en')
        response = l.lawchat(inp, qa_pipeline())
        response = translate(response,'en','ta')
    else:
        response = l.lawchat(inp, qa_pipeline())
    return response
    
@app.route('/get2', methods=['GET','POST'])
def get2():
    msg=request.form["msg"]
    inp=msg
    return get_resp_2(str(inp))
def get_resp_2(inp):
    context = scanned_image_to_text(file)
    chat = model1.start_chat(history=[])
    response = chatting(context, chat, inp)
    return response  

if __name__=="__main__":
    app.run()