from typing import Any, Optional, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, ActiveLoop
from rasa_sdk.types import DomainDict
from rasa_sdk.forms import FormValidationAction
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification
import torch
import requests
import logging

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Inizializzazione modelli 

embedder = SentenceTransformer("hkunlp/instructor-xl", device=device)


occupations = [
    "High school student", "University student", "PhD student", "Student", "Postdoc researcher",
    "Teacher", "Primary school teacher", "High school teacher", "University professor",
    "Academic researcher", "Private tutor", "School counselor", "Librarian",
    "Doctor", "General practitioner", "Surgeon", "Dentist", "Nurse", "Pharmacist",
    "Psychologist", "Psychiatrist", "Therapist", "Paramedic", "Medical assistant",
    "Veterinarian", "Healthcare administrator", "Radiologist", "Anesthesiologist",
    "Occupational therapist", "Speech therapist", "Nutritionist", "Dietitian",
    "Dental hygienist", "Medical technologist", "Lab technician", "Caregiver",
    "Hospice worker", "Home health aide",
    "Worker", "Freelancer", "Self-employed", "Part-time worker", "Intern", "Unemployed",
    "Looking for a job", "Homemaker", "Stay-at-home parent", "Retired", "Volunteer",
    "Software developer", "Frontend developer", "Backend developer", "Full stack developer",
    "Mobile developer", "Game developer", "DevOps engineer", "Machine learning engineer",
    "AI researcher", "Data scientist", "Data analyst", "IT support specialist",
    "System administrator", "Cybersecurity analyst", "Cloud architect", "Blockchain developer",
    "Game designer", "QA tester", "AR/VR developer", "Web designer", "Database administrator",
    "Tech blogger", "IT auditor",
    "Engineer", "Civil engineer", "Mechanical engineer", "Electrical engineer",
    "Industrial engineer", "Architect", "Construction worker", "Technician",
    "Mechanic", "Electrician", "Plumber", "Carpenter", "Blacksmith", "Locksmith",
    "Welder", "Roofer", "HVAC technician", "Elevator technician", "Surveyor",
    "Upholsterer", "Glazier", "Mason",
    "Truck driver", "Forklift operator", "Warehouse worker", "Logistics coordinator",
    "Supply chain manager", "Air traffic controller", "Pilot", "Flight attendant",
    "Ship captain", "Sailor", "Railway conductor", "Delivery driver", "Taxi driver",
    "Delivery coordinator", "Courier",
    "Artist", "Painter", "Illustrator", "Musician", "Composer", "Actor", "Filmmaker",
    "Photographer", "Video editor", "Graphic designer", "UX designer", "UI designer",
    "Fashion designer", "Interior designer", "Art director", "Set designer", "Animator",
    "Voice actor", "Fashion model", "Creative director", "Brand strategist", "Comic artist",
    "Screenwriter", "Music producer", "DJ", "Tattoo artist",
    "Journalist", "Writer", "Poet", "Content creator", "YouTuber", "Podcaster",
    "Influencer", "Social media manager",
    "Entrepreneur", "Business owner", "Startup founder", "Manager", "Team leader",
    "Project manager", "Product manager", "Salesperson", "Marketing specialist",
    "Financial analyst", "Accountant", "Bank teller", "HR specialist", "Consultant",
    "Business analyst", "Recruiter", "Investment banker", "Trader", "Venture capitalist",
    "Real estate agent", "Insurance agent", "Loan officer", "Auditor", "Economist",
    "Fundraiser", "Non-profit manager", "Executive assistant", "Office manager",
    "Administrative assistant", "Compliance officer", "Procurement specialist",
    "Risk analyst", "Operations manager", "Quality assurance specialist", "Event planner",
    "Scheduler",
    "Lawyer", "Paralegal", "Judge", "Legal assistant", "Compliance analyst", "Court clerk",
    "Legal advisor", "Mediator", "Notary public",
    "Police officer", "Firefighter", "Military personnel", "Public servant", "Politician",
    "Social worker", "Community organizer", "NGO worker", "Immigration officer",
    "City planner", "Diplomat", "Archivist", "Museum curator", "Census worker",
    "Chemist", "Biologist", "Physicist", "Environmental scientist", "Geologist",
    "Lab researcher", "Clinical researcher", "Science communicator", "Statistician",
    "Farmer", "Agricultural engineer", "Ecologist", "Beekeeper", "Fisherman",
    "Forestry worker", "Landscape designer", "Environmental consultant", "Park ranger",
    "Zookeeper",
    "Customer service agent", "Waiter", "Barista", "Chef", "Cook", "Cashier",
    "Retail worker", "Janitor", "Security guard", "Bartender",
    "Babysitter", "Pet sitter", "Dog walker", "Housekeeper", "Personal trainer",
    "Fitness coach", "Yoga instructor", "Life coach", "Motivational speaker",
    "Spiritual advisor", "Psychic", "Magician", "Model", "Escort", "Club promoter",
    "Event host", "Auctioneer",
    "Special education teacher", "Education consultant", "Curriculum developer",
    "School principal", "Academic advisor", "Test prep tutor", "Language teacher",
    "Early childhood educator", "Training specialist",
    "Student (generic)", "Job seeker", "Gig worker", "Remote worker", "Digital nomad",
    "No occupation", "Prefer not to say", 
]

instruction_occ = "Occupation category:"
occupation_embeddings = embedder.encode([[instruction_occ, occ] for occ in occupations], convert_to_tensor=True)

def classify_occupations_instructor(user_input, threshold=0.4, top_k=None):

    user_embedding = embedder.encode(
        [["What are the occupations of this person?:", user_input]], 
        convert_to_tensor=True
    )[0]

    cosine_scores = util.cos_sim(user_embedding, occupation_embeddings)[0]

    occupation_score_pairs = [
        (occupations[i], float(score)) 
        for i, score in enumerate(cosine_scores) 
        if score >= threshold
    ]

    occupation_score_pairs.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        occupation_score_pairs = occupation_score_pairs[:top_k]

    return [label for label, score in occupation_score_pairs] if occupation_score_pairs else ["Other"]





interests = [
    "running", "jogging", "walking", "cycling", "swimming", "hiking", "climbing",
    "football", "soccer", "basketball", "tennis", "volleyball", "skiing", "snowboarding",
    "skating", "surfing", "martial arts", "boxing", "gym", "fitness", "yoga",
    "pilates", "aerobics", "dance fitness", "crossfit", "bodybuilding",
    "listening to music", "playing instruments", "singing", "composing music",
    "attending concerts", "music production", "DJing", "karaoke", "classical music",
    "rock music", "pop music", "jazz", "hip hop", "electronic music",
    "reading fiction", "reading non-fiction", "science fiction", "fantasy books",
    "mystery novels", "philosophy books", "self-help books", "poetry",
    "writing stories", "blogging", "journaling", "writing poetry", "creative writing",
    "video games", "mobile games", "MMORPGs", "strategy games", "board games",
    "card games", "chess", "Dungeons and Dragons", "puzzle games", "game development",
    "drawing", "painting", "sculpting", "digital art", "graphic design", "calligraphy",
    "photography", "film making", "video editing", "animation", "fashion design",
    "makeup art", "interior design", "crafting", "origami", "knitting", "sewing",
    "science", "physics", "astronomy", "biology", "chemistry", "mathematics", "philosophy",
    "psychology", "history", "politics", "geography", "languages", "learning new skills",
    "debating", "TED Talks", "documentaries", "museums", "archaeology",
    "cooking", "baking", "trying new recipes", "street food", "vegetarian food",
    "vegan cooking", "wine tasting", "coffee brewing", "craft beer",
    "traveling", "backpacking", "road trips", "exploring cities", "cultural exchange",
    "camping", "van life", "travel blogging", "airbnb experiences",
    "yoga", "meditation", "mindfulness", "journaling", "sleep optimization",
    "minimalism", "self-care", "productivity", "personal development",
    "gardening", "plants", "birdwatching", "fishing", "hunting", "camping",
    "forests", "mountains", "beaches", "animals", "pets", "dog walking",
    "volunteering at shelters", "horseback riding",
    "DIY projects", "woodworking", "home improvement", "electronics repair",
    "model building", "mechanics", "robotics", "3D printing",
    "coding", "web development", "AI and machine learning", "tech news",
    "mobile apps", "gadget reviews", "cybersecurity", "hacking", "Linux",
    "open source", "startups", "digital marketing", "crypto", "NFTs",
    "volunteering", "activism", "environmental causes", "human rights",
    "religion", "spirituality", "astrology", "parenting", "family time",
    "socializing", "meeting new people", "clubbing", "networking", "nothing"
]

instruction_int = "Represent this interest category:"
interest_embeddings = embedder.encode([[instruction_int, int] for int in interests], convert_to_tensor=True)

def classify_interests_instructor(user_input, threshold=0.4, top_k=None):

    user_embedding = embedder.encode(
        [["Represent the interests of this person:", user_input]], 
        convert_to_tensor=True
    )[0]

    cosine_scores = util.cos_sim(user_embedding, interest_embeddings)[0]

    interest_score_pairs = [
        (interests[i], float(score)) 
        for i, score in enumerate(cosine_scores) 
        if score >= threshold
    ]

    interest_score_pairs.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        interest_score_pairs = interest_score_pairs[:top_k]

    return [label for label, score in interest_score_pairs] if interest_score_pairs else ["Other"]





common_health_labels_en = [
    "anxiety", "depression", "stress", "insomnia", "low self-esteem", "panic attacks",
    "burnout", "loneliness", "obsessive-compulsive disorder (OCD)", "post-traumatic stress disorder (PTSD)",
    "social anxiety disorder", "bipolar disorder", "borderline personality disorder", "repressed anger", 
    "procrastination", "adjustment disorder", "emotional dependency", "binge eating disorder", 
    "emotional eating", "relationship difficulties", "obesity", "overweight", "anorexia", "hypertension", 
    "diabetes", "high cholesterol", "thyroid problems", "chronic back pain", "joint pain", "asthma",
    "heart problems", "chronic pain", "poor nutrition", "physical inactivity", "tooth decay", "migraine", 
    "persistent fatigue", "metabolic syndrome", "food intolerances", "chronic stress", "substance abuse", 
    "alcoholism", "smoking", "drug addiction", "gambling addiction", "internet addiction", 
    "social media addiction", "good overall health", "no major health problems"
]

intruction_health = "Represent this health condition category:"
health_embeddings = embedder.encode([[intruction_health, label] for label in common_health_labels_en], convert_to_tensor=True)

def classify_health_condition_instructor(user_input, threshold=0.8, top_k=None):

    user_embedding = embedder.encode(
        [["Represent the health condition of this person:", user_input]], 
        convert_to_tensor=True
    )[0]

    cosine_scores = util.cos_sim(user_embedding, health_embeddings)[0]

    health_score_pairs = [
        (common_health_labels_en[i], float(score)) 
        for i, score in enumerate(cosine_scores) 
        if score >= threshold
    ]

    health_score_pairs.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        health_score_pairs = health_score_pairs[:top_k]

    return [label for label, score in health_score_pairs] if health_score_pairs else ["Other"]




lifestyle_labels = [
    "sedentary", "active", "busy", "stressful", "healthy", 
    "unhealthy", "irregular","balanced", "routine", "unstructured"
]


intruction_lifestyle = "Lifestyle category:"
lifestyle_embeddings = embedder.encode([[intruction_lifestyle, label] for label in lifestyle_labels], convert_to_tensor=True)

def classify_lifestyle_instructor(user_input, threshold=0.8, top_k=None):

    user_embedding = embedder.encode(
        [["Represent the lifestyle of this person:", user_input]], 
        convert_to_tensor=True
    )[0]

    cosine_scores = util.cos_sim(user_embedding, lifestyle_embeddings)[0]

    lifestyle_score_pairs = [
        (lifestyle_labels[i], float(score)) 
        for i, score in enumerate(cosine_scores) 
        if score >= threshold
    ]

    lifestyle_score_pairs.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        lifestyle_score_pairs = lifestyle_score_pairs[:top_k]

    return [label for label, score in lifestyle_score_pairs] if lifestyle_score_pairs else ["Other"]



classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device,
    use_fast=True
)            

candidate_labels = [
    "anxiety", "stress", "nutrition", "physical activity", "weight",
    "medication adherence", "sleep", "smoking", "alcohol", "relationships",
    "motivation", "addictions", "self-esteem", "insecurity"
]

  

# Azioni personalizzate per Rasa

ner_tokenizer = AutoTokenizer.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
ner_model = AutoModelForTokenClassification.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")


class ActionExtractName(Action):
    def name(self) -> Text:
        return "action_extract_name"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text").strip()

        # Step 1: Prova con NER
        entities = ner_pipeline(user_message)
        name = None

        for ent in entities:
            if ent["entity_group"] == "PER":
                name = ent["word"]
                break

        # Step 2: Fallback se NER non trova nulla
        if not name:
            if user_message.istitle() and " " not in user_message:
                # singola parola con iniziale maiuscola, potrebbe essere un nome
                name = user_message

        if name:
            dispatcher.utter_message(text=f"Nice to meet you, {name}!")
            return [SlotSet("name", name)]
        else:
            dispatcher.utter_message(text="Sorry, I couldn’t catch your name.")
            return []



class ActionClassifyTalkType(Action):

    def name(self) -> Text:
        return "action_classify_talk_type"
    
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("./best_model")
        self.model = BertForSequenceClassification.from_pretrained("./best_model")
        self.model.eval()
    
    def run(self, dispatcher, tracker, domain):

        user_message = tracker.latest_message.get("text")

        model_inputs = self.tokenizer(
        user_message, 
        truncation=True, 
        padding="max_length", 
        max_length=128, 
        return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**model_inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()
            prob = torch.softmax(logits, dim=1)[0][predicted_label].item()

        if predicted_label == 0:
            predicted_str = "change"
            dispatcher.utter_message(text="I'm glad you're motivated to change!")
        else:
            predicted_str = "sustain"
            dispatcher.utter_message(text="I understand, sometimes change can feel hard to face.")


        logger.info(f"text: {user_message}")
        logger.info(f"Label: {predicted_str}")
        logger.info(f"Confidence: {prob:.2f}")

        return[SlotSet("talk_type", predicted_str)]
    



class ActionClassifyUserOccupation(Action):

    def name(self) -> Text:
        return "action_classify_user_occupation"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text")

        occupation = classify_occupations_instructor(user_message, threshold=0.8, top_k=2)

        
        return [SlotSet("occupation", occupation)]
    

class ActionClassifyUserInterests(Action):

    def name(self) -> Text:
        return "action_classify_user_interests"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text")

        selected_interests = classify_interests_instructor(user_message, threshold=0.8, top_k=3)

        
        return [SlotSet("interests", selected_interests)]



class ActionClassifyUserHealthCondition(Action):

    def name(self) -> Text:
        return "action_classify_user_health_condition"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text")

        selected_health_labels = classify_health_condition_instructor(user_message, threshold=0.8, top_k=3)

        
        return [SlotSet("health_condition", selected_health_labels)]



class ActionClassifyUserLifestyle(Action):

    def name(self) -> Text:
        return "action_classify_user_lifestyle"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text")

        selected_lifestyle_labels = classify_lifestyle_instructor(user_message, threshold=0.8, top_k=2)

        
        return [SlotSet("lifestyle", selected_lifestyle_labels)]





class ActionClassifyTopic(Action):

    def name(self) -> Text:
        return "action_classify_topic"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text")

        topics = tracker.get_slot("topic_list") or []
     
        result = classifier(user_message, candidate_labels, multi_label=True)
        topic = result["labels"][0]
        score = result["scores"][0]

        if score > 0.8:
            
            if topic not in topics:
                updated_topics = topics + [topic]
            else: 
                updated_topics = topics

            return [
                SlotSet("topic_list", updated_topics),
                SlotSet("current_topic", topic)]
        else:
            dispatcher.utter_message(text="I'm not sure I understand the problem, could you explain it a bit more?")
            return []





# class ActionLLMFallback(Action):

#     def name(self) -> Text:
#         return "action_llm_fallback"
    
#     def classify_topic(self, message: str) -> Optional[str]:
#         result = classifier(message, candidate_labels, multi_label=True)
#         if result['scores'][0] > 0.8:
#             return result['labels'][0]
#         return None
    
#     def get_model(self) -> str:

#         try:
#             response = requests.get("http://localhost:1234/v1/models")
#             if response.status_code == 200:
#                 models = response.json().get("data", [])
#                 if models:
#                     return models[0]["id"] 
#         except Exception as e:
#             print(f"Error: {e}")
        
#         return "mistral-7b-instruct-v0.3"
     
#     def run(self, 
#             dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
#         user_message = tracker.latest_message.get('text')

#         topic = self.classify_topic(user_message)
     
#         if topic:
#             current_list = tracker.get_slot("topic_list") or []
#             if topic not in current_list:
#                 updated_topics = current_list + [topic]
#                 return [
#                     SlotSet("current_topic", topic),
#                     SlotSet("topic_list", updated_topics)
#                 ]
#             else:
#                 return [SlotSet("current_topic", topic)]



#         model_name = self.get_model()
        
#         headers = {
#             "Content-Type": "application/json"
#         }
#         payload = {
#             "model": model_name,  
#             "messages": [
#                 {"role": "user", 
#                  "content": user_message}
#             ],
#             "temperature": 0.5
#         }

#         response = requests.post("http://localhost:1234/v1/chat/completions", json=payload, headers=headers)
        
#         if response.status_code == 200:
#             reply = response.json()['choices'][0]['message']['content']
#         else:
#             reply = "I'm sorry, an error occurred."

#         dispatcher.utter_message(text=reply)
#         return []
    





class ActionConversationStarted(Action):

    def name(self) -> Text:
        return "action_conversation_started"
    
    def run(self, dispatcher, tracker, domain):
        return[SlotSet("conversation_started", True)]
    

    



class ValidateUserInfoForm(FormValidationAction):

    def name(self) -> Text:
        return "validate_user_info_form"
    
    def validate_name(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:

        
        if isinstance(slot_value, str) and slot_value.strip():
            return {"name": slot_value.strip()}
        else:
            dispatcher.utter_message(text="Please insert a valid name.")
            return {"name": None}
            
    def validate_age(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:

        try:
            age = int(slot_value)
            if 0 < age < 120:
                return {"age": age}
            else:
                dispatcher.utter_message(text="Age must be a number between 0 and 120.")
                return {"age": None}
        except Exception:
            dispatcher.utter_message(text="Age must be a number.")
            return {"age": None}



class ActionSubmitFormUserInfo(Action):

    def name(self) -> str:
        return "action_submit_user_info_form"

    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: dict):
        
        name = tracker.get_slot("name")
        dispatcher.utter_message(text=f"Thanks {name}! I've saved your information. Tell me more about how you're feeling today.")

        
        return [
            SlotSet("form_completed", True),
            ActiveLoop(None)
        ]
    
