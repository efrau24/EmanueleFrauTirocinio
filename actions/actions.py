from typing import Any, Optional, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, ActiveLoop, FollowupAction
from rasa_sdk.types import DomainDict
from rasa_sdk.forms import FormValidationAction
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification
import torch
import requests
import logging
import json 
import re

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Inizializzazione modelli 

embedder = SentenceTransformer("hkunlp/instructor-xl", device=device)

occupations = [
    # Students / Education
    "High school student", "University student", "Postgraduate student",
    "Teacher", "School teacher", "University professor", "Academic researcher",
    "Private tutor", "School counselor", "Librarian", "Education consultant",
    "Curriculum developer", "School principal", "Academic advisor", "Language teacher",
    "Early childhood educator", "Special education teacher", "Training specialist",
    
    # Healthcare
    "Doctor", "Dentist", "Nurse", "Pharmacist", "Psychologist", "Psychiatrist",
    "Therapist", "Medical assistant", "Paramedic", "Veterinarian",
    "Healthcare administrator", "Radiologist", "Anesthesiologist",
    "Occupational therapist", "Speech therapist", "Nutritionist", "Dietitian",
    "Dental hygienist", "Medical technologist", "Lab technician",
    "Caregiver", "Hospice worker", "Home health aide",

    # Generic work statuses
    "Worker", "Freelancer", "Self-employed", "Part-time worker", "Intern",
    "Unemployed", "Job seeker", "Homemaker", "Stay-at-home parent",
    "Retired", "Volunteer", "Gig worker", "Remote worker", "Digital nomad",

    # Tech & IT
    "Software developer", "Full stack developer", "Mobile developer", 
    "Game developer", "DevOps engineer", "Machine learning engineer",
    "AI researcher", "Data scientist", "Data analyst", "IT support specialist",
    "System administrator", "Cybersecurity analyst", "Cloud architect",
    "Blockchain developer", "Game designer", "QA tester", "Web designer",
    "UI/UX designer", "Database administrator", "IT auditor", "Tech blogger",

    # Engineering & Technical
    "Engineer", "Civil engineer", "Mechanical engineer", "Electrical engineer",
    "Industrial engineer", "Architect", "Construction worker", "Technician",
    "Mechanic", "Electrician", "Plumber", "Carpenter", "Welder", "Roofer",
    "HVAC technician", "Surveyor", "Glazier", "Mason",

    # Transport & Logistics
    "Truck driver", "Forklift operator", "Warehouse worker", "Logistics coordinator",
    "Supply chain manager", "Air traffic controller", "Pilot", "Flight attendant",
    "Ship captain", "Railway conductor", "Delivery driver", "Taxi driver",
    "Courier",

    # Art & Media
    "Artist", "Painter", "Illustrator", "Musician", "Composer", "Actor",
    "Filmmaker", "Photographer", "Video editor", "Graphic designer",
    "Fashion designer", "Interior designer", "Art director", "Animator",
    "Voice actor", "Model", "Creative director", "Comic artist", "Screenwriter",
    "Music producer", "DJ", "Tattoo artist",

    # Communication & Content
    "Journalist", "Writer", "Poet", "Content creator", "YouTuber",
    "Podcaster", "Influencer", "Social media manager",

    # Business & Management
    "Entrepreneur", "Business owner", "Startup founder", "Manager",
    "Project manager", "Product manager", "Salesperson", "Marketing specialist",
    "Financial analyst", "Accountant", "HR specialist", "Consultant",
    "Business analyst", "Recruiter", "Investment banker", "Trader",
    "Real estate agent", "Insurance agent", "Loan officer", "Auditor",
    "Economist", "Fundraiser", "Non-profit manager", "Executive assistant",
    "Office manager", "Administrative assistant", "Compliance officer",
    "Procurement specialist", "Operations manager", "Quality assurance specialist",
    "Event planner",

    # Legal
    "Lawyer", "Paralegal", "Judge", "Legal assistant", "Court clerk",
    "Legal advisor", "Mediator", "Notary public",

    # Public sector & Safety
    "Police officer", "Firefighter", "Military personnel", "Public servant",
    "Politician", "Social worker", "Community organizer", "NGO worker",
    "Immigration officer", "City planner", "Diplomat",

    # Science & Environment
    "Scientist", "Chemist", "Biologist", "Physicist", "Environmental scientist",
    "Geologist", "Lab researcher", "Clinical researcher", "Statistician",
    "Science communicator", "Ecologist", "Environmental consultant",
    "Agricultural engineer", "Forestry worker", "Park ranger", "Zookeeper",
    "Farmer", "Fisherman", "Beekeeper", "Landscape designer",

    # Service & Hospitality
    "Customer service agent", "Waiter", "Chef", "Cashier",
    "Retail worker", "Janitor", "Security guard", "Bartender",

    # Personal care & Lifestyle
    "Babysitter", "Pet sitter", "Dog walker", "Housekeeper", "Personal trainer",
    "Fitness coach", "Yoga instructor", "Life coach", "Motivational speaker",
    "Spiritual advisor", "Psychic",

    # Events & Entertainment
    "Magician", "Escort", "Club promoter", "Event host", "Auctioneer",

    # Neutral
    "No occupation", "Prefer not to say"
]

instruction_occ = "Represent the occupation mentioned in this sentence:"
occupation_embeddings = embedder.encode([[instruction_occ, occ] for occ in occupations], convert_to_tensor=True)

def classify_occupations_instructor(user_input, threshold=0.8, top_k=None):

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



# === Lista di interessi ===
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
    "volunteering at shelters", "horseback riding", "electronics",
    "DIY projects", "woodworking", "home improvement", "electronics repair",
    "model building", "mechanics", "robotics", "3D printing",
    "coding", "web development", "AI and machine learning", "tech news",
    "mobile apps", "gadget reviews", "cybersecurity", "hacking", "Linux",
    "open source", "startups", "digital marketing", "crypto", "NFTs",
    "volunteering", "activism", "environmental causes", "human rights",
    "religion", "spirituality", "astrology", "parenting", "family time",
    "socializing", "meeting new people", "clubbing", "networking", "nothing"
]

# === Macro categorie ===
macro_categories = {
    "Fitness & Sports": [
        "running", "jogging", "walking", "cycling", "swimming", "hiking", "climbing",
        "football", "soccer", "basketball", "tennis", "volleyball", "skiing", "snowboarding",
        "skating", "surfing", "martial arts", "boxing", "gym", "fitness", "yoga", "pilates",
        "aerobics", "dance fitness", "crossfit", "bodybuilding"
    ],
    "Music": [
        "listening to music", "playing instruments", "singing", "composing music",
        "attending concerts", "music production", "DJing", "karaoke", "classical music",
        "rock music", "pop music", "jazz", "hip hop", "electronic music"
    ],
    "Literature": [
        "reading fiction", "reading non-fiction", "science fiction", "fantasy books",
        "mystery novels", "philosophy books", "self-help books", "poetry",
        "writing stories", "blogging", "journaling", "writing poetry", "creative writing"
    ],
    "Gaming": [
        "video games", "mobile games", "MMORPGs", "strategy games", "board games",
        "card games", "chess", "Dungeons and Dragons", "puzzle games", "game development"
    ],
    "Arts": [
        "drawing", "painting", "sculpting", "digital art", "graphic design", "calligraphy",
        "photography", "film making", "video editing", "animation", "fashion design",
        "makeup art", "interior design", "crafting", "origami", "knitting", "sewing"
    ],
    "Science & Education": [
        "science", "physics", "astronomy", "biology", "chemistry", "mathematics", "philosophy",
        "psychology", "history", "politics", "geography", "languages", "learning new skills",
        "debating", "TED Talks", "documentaries", "museums", "archaeology"
    ],
    "Food & Cooking": [
        "cooking", "baking", "trying new recipes", "street food", "vegetarian food",
        "vegan cooking", "wine tasting", "coffee brewing", "craft beer"
    ],
    "Travel & Adventure": [
        "traveling", "backpacking", "road trips", "exploring cities", "cultural exchange",
        "camping", "van life", "travel blogging", "airbnb experiences"
    ],
    "Well-being & Lifestyle": [
        "yoga", "meditation", "mindfulness", "journaling", "sleep optimization",
        "minimalism", "self-care", "productivity", "personal development"
    ],
    "Nature & Outdoors": [
        "gardening", "plants", "birdwatching", "fishing", "hunting", "camping",
        "forests", "mountains", "beaches", "animals", "pets", "dog walking",
        "volunteering at shelters", "horseback riding"
    ],
    "Tech & Engineering": [
        "electronics", "DIY projects", "woodworking", "home improvement", "electronics repair",
        "model building", "mechanics", "robotics", "3D printing", 
        "coding", "web development", "AI and machine learning", "tech news",
        "mobile apps", "gadget reviews", "cybersecurity", "hacking", "Linux",
        "open source", "startups", "digital marketing", "crypto", "NFTs"
    ],
    "Social & Humanitarian": [
        "volunteering", "activism", "environmental causes", "human rights",
        "religion", "spirituality", "astrology", "parenting", "family time",
        "socializing", "meeting new people", "clubbing", "networking"
    ],
    "Other": ["nothing"]
}

# === Mappatura da interesse a macro-categoria ===
interest_to_macro = {
    interest: macro for macro, interest_list in macro_categories.items()
    for interest in interest_list
}

instruction_int = "Represent this interest category:"
interest_embeddings = embedder.encode([[instruction_int, int] for int in interests], convert_to_tensor=True)

def classify_interests_instructor(user_input, threshold=0.4, top_k=None):

    user_embedding = embedder.encode(
        [["What are the interests of this person?:", user_input]], 
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

   
def classify_interests_with_macro(user_input, threshold=0.4, top_k=None):
    fine_labels = classify_interests_instructor(user_input, threshold=threshold, top_k=top_k)

    macro_set = set()
    for label in fine_labels:
        macro = interest_to_macro.get(label, "Other")
        macro_set.add(macro)

    return {
        "fine_labels": fine_labels,
        "macro_labels": list(macro_set)
    } 
    



common_health_labels_en = [
    "anxiety", "depression", "stress", "insomnia", "low self-esteem",
    "panic attacks", "burnout", "loneliness", "ocd", "ptsd",
    "drug addiction", "alcoholism", "smoking", "gambling addiction",
    "internet addiction", "social media addiction", "binge eating",
    "obesity", "anorexia", "poor nutrition", "lifestyle issues",
    "low motivation"
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




# Azioni personalizzate per Rasa

ner_tokenizer = AutoTokenizer.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
ner_model = AutoModelForTokenClassification.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")



    
def get_model() -> str:
    try:
        response = requests.get("http://localhost:1234/v1/models")
        if response.status_code == 200:
            models = response.json().get("data", [])
            if models:
                return models[0]["id"]
    except Exception as e:
        print(f"Error: {e}")
    return "mistral-7b-instruct-v0.3"



class ActionExtractName(Action):
    def name(self) -> Text:
        return "action_extract_name"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text").strip()

        entities = ner_pipeline(user_message)
        name = None

        for ent in entities:
            if ent["entity_group"] == "PER":
                name = ent["word"]
                break


        if not name:
            if user_message.istitle() and " " not in user_message:

                name = user_message

        if name:
            dispatcher.utter_message(text=f"Nice to meet you, {name}!")
            return [SlotSet("name", name)]
        else:
            dispatcher.utter_message(text="Sorry, I couldn’t catch your name.")
            return []



class ActionClassifyUserOccupation(Action):

    def name(self) -> Text:
        return "action_classify_user_occupation"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Controllo se lo slot è già pieno
        current_value = tracker.get_slot("occupation")
        if current_value:
            return []  # non sovrascrivo

        user_message = tracker.latest_message.get("text")

        return [SlotSet("occupation", user_message)]



class ActionClassifyUserInterests(Action):

    def name(self) -> Text:
        return "action_classify_user_interests"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Controllo se lo slot è già pieno
        current_value = tracker.get_slot("interests")
        if current_value:
            return []  # non sovrascrivo

        user_message = tracker.latest_message.get("text")
        results = classify_interests_with_macro(user_message, threshold=0.8, top_k=3)

        fine_labels = results["fine_labels"]
        macro_labels = results["macro_labels"]

        return [
            SlotSet("interests", macro_labels),
            SlotSet("sub_interests", fine_labels)
        ]



# class ActionClassifyUserHealthCondition(Action):

#     def name(self) -> Text:
#         return "action_classify_user_health_condition"
    
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

#         # Controllo se lo slot è già pieno
#         current_value = tracker.get_slot("health_condition")
#         if current_value:
#             return [FollowupAction("action_extract_issues")]

#         user_message = tracker.latest_message.get("text")

#         return [SlotSet("health_condition", user_message)]


# class ActionConversationStarted(Action):

#     def name(self) -> Text:
#         return "action_conversation_started"
    
#     def run(self, dispatcher, tracker, domain):
#         return[SlotSet("conversation_started", True)]
    


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

    def build_prompt(self, name, age, occupation, interests) -> str:
        return f"""You are an empathetic mental health support chatbot. The user has just told you a bit about themselves through a short exchange.

        Here is what they've shared:

        - Name: {name}
        - Age: {age}
        - Occupation: {occupation}
        - Interests: {interests}

        Now, write a short, warm, and natural-sounding message that shows you’ve understood their situation. Your message should:

        - Reflect back something meaningful they’ve shared (e.g., age, job, lifestyle, interests…)
        - Make them feel heard and understood
        - Finish with "How can i help you today?" to invite them to share more.
        
        Keep the tone friendly, non-judgmental, and supportive. Do not introduce yourself, greet, repeat your role or talk about yourself — just continue the conversation as if you already know each other."""

    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        name = tracker.get_slot("name")
        age = tracker.get_slot("age")
        occupation = tracker.get_slot("occupation")
        interests = tracker.get_slot("interests")

        prompt = self.build_prompt(name, age, occupation, interests)

        model = get_model()
        headers = { "Content-Type": "application/json" }
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        try:
            response = requests.post("http://localhost:1234/v1/chat/completions", json=payload, headers=headers)
            if response.status_code == 200:
                reply = response.json()['choices'][0]['message']['content']
            else:
                reply = "Thanks for completing the form! If you’d like, feel free to tell me if there’s anything you’d like to work on or explore together."
        except Exception as e:
            print(f"Error: {e}")
            reply = "Thanks for completing the form! If you’d like, feel free to tell me if there’s anything you’d like to work on or explore together."

        dispatcher.utter_message(text=reply)
        

        
        return [
            SlotSet("form_completed", True),
            ActiveLoop(None)
        ]
    

# --- 1. Extraction of issues from user messages ---
class ActionExtractIssues(Action):
    def name(self) -> Text:
        return "action_extract_issues"
    
    #         1. Write a short, warm, natural-sounding reply that shows understanding.

    def build_prompt(self, message) -> str:
        return f"""You are an empathetic health support chatbot. 
        The user has just told you something about their mental health or lifestyle.

        Here is what they've shared: "{message}"

        Your tasks:
        Identify ALL relevant issues in their message and classify them into one or more of these categories:
        - anxiety
        - depression
        - stress
        - insomnia
        - low self-esteem
        - panic attacks
        - burnout
        - loneliness
        - OCD
        - PTSD
        - drug addiction
        - alcoholism
        - smoking
        - gambling addiction
        - internet addiction
        - social media addiction
        - binge eating
        - obesity
        - anorexia
        - poor nutrition
        - lifestyle issues

        If the issues seems realted, include only the most relevant one. If no issues are mentioned, return an empty list.

        Return your answer strictly in this JSON format:
        {{
        "labels": ["<category1>", "<category2>", ...]
        }}"""

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        last_message = tracker.latest_message.get("text", "")
        if not last_message:
            return []

        prompt = self.build_prompt(last_message)
        model = get_model()
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5
        }

        try:
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                json=payload,
                headers=headers
            )
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                try:
                    data = json.loads(content)
                    labels = data.get("labels", [])
                    if not isinstance(labels, list):
                        labels = [labels]
                except Exception as e:
                    print(f"JSON parsing error: {e}")
                    labels = []
            else:
                
                labels = []

        except Exception as e:
            print(f"Error: {e}")
            labels = []

        

        old_issues = tracker.get_slot("issues_profile") or {}
        new_issues = {label: {} for label in labels}
        merged_issues = {**old_issues, **new_issues}

        return [SlotSet("issues_profile", merged_issues)]
    


class ActionAskFocusIssue(Action):
    def name(self) -> Text:
        return "action_ask_focus_issue"
    
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        issues_profile  = tracker.get_slot("issues_profile") or {}
        pending_issues = []

        # Normalizza in dict per sicurezza
        if isinstance(issues_profile, list):
            issues_profile = {i: {} for i in issues_profile}
        
        for issue, details in issues_profile.items():
            if not details:  
                pending_issues.append(issue)

        if not pending_issues:     
            dispatcher.utter_message(
                text="Now, if there's anything else you'd like to discuss or explore, feel free to let me know!"
            )
            return []
        elif len(pending_issues) == 1:
            issue = pending_issues[0]
            dispatcher.utter_message(
                text=f"It sounds like you might be dealing with {issue}. Let's focus on that for now."
            )
            return [
                SlotSet("current_issue", issue),
                FollowupAction(name="action_activate_form")
            ]
        else:
            issues_list = ", ".join(pending_issues)
            dispatcher.utter_message(
                text=(
                    f"It sounds like there are several things on your mind: {issues_list}. "
                    f"Which one feels most important to focus on right now?"
                )
            )
                
        return []

class ActionSetCurrentIssue(Action):
    def name(self) -> str:
        return "action_set_current_issue"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        user_text = tracker.latest_message.get("text")

        # Assumo che la funzione ritorni lista -> prendo primo
        issue = classify_health_condition_instructor(user_text, threshold=0.8, top_k=1)
        if isinstance(issue, list):
            issue = issue[0] if issue else None

        issues_profile = tracker.get_slot("issues_profile") or {}

        if isinstance(issues_profile, dict):
            remaining_issues = {k: v for k, v in issues_profile.items() if k != issue}
        elif isinstance(issues_profile, list):
            remaining_issues = [i for i in issues_profile if i != issue]
        else:
            remaining_issues = []

        return [
            SlotSet("current_issue", issue),
            SlotSet("remaining_issues", remaining_issues),
            FollowupAction(name="action_activate_form")
        ]


form_mapping = {
            # Mood & mental health
            "anxiety": "mood_form",
            "depression": "mood_form",
            "stress": "mood_form",
            "panic attacks": "mood_form",
            "burnout": "mood_form",
            "loneliness": "mood_form",
            "ocd": "mood_form",
            "ptsd": "mood_form",
            "low self-esteem": "self_esteem_form",

            # Sleep
            "insomnia": "sleep_form",

            # Addictions
            "drug addiction": "addiction_form",
            "alcoholism": "addiction_form",
            "smoking": "addiction_form",
            "gambling addiction": "addiction_form",
            "internet addiction": "addiction_form",
            "social media addiction": "addiction_form",

            # Nutrition & lifestyle
            "binge eating": "nutrition_form",
            "obesity": "nutrition_form",
            "anorexia": "nutrition_form",
            "poor nutrition": "nutrition_form",
            "lifestyle issues": "lifestyle_form",
            "low motivation": "motivation_form",
        }

class ActionActivateForm(Action):
    def name(self) -> str:
        return "action_activate_form"
    
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        current_issue = tracker.get_slot("current_issue")

        if isinstance(current_issue, list):
            current_issue = current_issue[0] if current_issue else None

        if not current_issue:
            return []

        current_issue = current_issue.lower()
        form_to_activate = form_mapping.get(current_issue)

        if not form_to_activate:
            return []

       
        if tracker.active_loop.get('name') == form_to_activate:
            return []

        
        return [ActiveLoop(name=form_to_activate),
                FollowupAction(name=form_to_activate)]
    

class ActionSaveIssueProfile(Action):
    def name(self) -> Text:
        return "action_save_issue_profile"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        current_issue = tracker.get_slot("current_issue")
        issues_profile = tracker.get_slot("issues_profile") or {}

        if isinstance(current_issue, list):
            current_issue = current_issue[0] if current_issue else None

        if not current_issue:
            return []


        
        form_slots = {
            "mood_form": ["symptom_description", "duration", "triggers", "impact", "coping_strategies"],
            "sleep_form": ["sleep_pattern", "sleep_quality", "sleep_duration", "bedtime_routine", "sleep_impact"],
            "intrusive_thoughts_form": ["intrusive_thoughts_description", "frequency", "intensity", "avoidance", "intrusive_thoughts_impact"],
            "addiction_form": ["substance_or_behavior", "addiction_frequency", "addiction_triggers", "attempts_to_quit", "addiction_impact"],
            "nutrition_form": ["eating_habits", "body_image", "nutrition_physical_activity", "nutrition_duration", "nutrition_impact"],
            "motivation_form": ["motivation_level", "barriers", "goals", "support_system", "strategies_to_increase_motivation"],
            "self_esteem_form": ["self_perception", "negative_self_talk", "social_interactions", "achievements", "self_esteem_coping_strategies"],
            "lifestyle_form": ["daily_routine", "lifestyle_physical_activity", "sleep_habits", "lifestyle_nutrition", "stress_management"],
        }

        
        form_name = form_mapping.get(current_issue)
        if not form_name:
            return []

        slots_to_save = form_slots.get(form_name, [])

        
        issue_data = {
            slot: tracker.get_slot(slot)
            for slot in slots_to_save
            if tracker.get_slot(slot) is not None
        }
        
        existing_data = issues_profile.get(current_issue, {})
        
        updated_data = {**existing_data, **issue_data}

        issues_profile[current_issue] = updated_data

        return [SlotSet("issues_profile", issues_profile)]


    
# --- 2. Updating emotional state (mood) ---
class ActionUpdateMood(Action):

    def name(self) -> Text:
        return "action_update_mood"


    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        recent_message = tracker.latest_message.get("text")
        if not recent_message:
            return []

        prompt = f"""
        Analyze the message and return the current emotional state:
        \"\"\"{recent_message}\"\"\"
        
        Reply in JSON:
        {{
            "mood_general": "positive|neutral|negative",
            "emotion_dominant": "joy|sadness|fear|anger|calm|hope|etc",
            "energy_level": "low|medium|high"
        }}
        """

        try:
            model = get_model()
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 150,
                    "temperature": 0.2
                }
            )
            text = response.json()["choices"][0]["text"].strip()
            json_data = json.loads(text[text.find("{"):text.rfind("}")+1])
        except Exception as e:
            dispatcher.utter_message(text=f"Error parsing mood: {e}")
            return []

        return [
            SlotSet("mood_general", json_data.get("mood_general", "neutral")),
            SlotSet("emotion_dominant", json_data.get("emotion_dominant", "calm")),
            SlotSet("energy_level", json_data.get("energy_level", "medium")),
        ]


# --- 3. Personality profile (Big Five) ---

class ActionUpdatePersonality(Action):
    def name(self) -> Text:
        return "action_update_personality"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # prendi tutti i messaggi dell'utente
        all_messages = [e.get("text") for e in tracker.events if e.get("event") == "user"]
        if len(all_messages) < 5:  
            return []

        profile_text = "\n".join(all_messages)

        prompt = f"""
                Analyze the following conversation and return the Big Five personality traits.

                Conversation:
                \"\"\"{profile_text}\"\"\"

                Reply ONLY with valid JSON (no explanation, no extra text). 
                Format:
                {{
                "extraversion": <number 0-10>,
                "emotional_stability": <number 0-10>,
                "openness": <number 0-10>,
                "agreeableness": <number 0-10>,
                "conscientiousness": <number 0-10>
                }}
                """

        try:
            model = get_model()
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "You are an assistant that ONLY replies with valid JSON.\n\n"
                                + prompt
                            )
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 200
                }
            )
            

            text = response.json()["choices"][0]["message"]["content"].strip()

            # estrai solo il blocco JSON con regex
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in model response")
            
            json_data = json.loads(match.group())

        except Exception as e:
            dispatcher.utter_message(text=f"Error parsing personality: {e}")
            return []

        # funzione di smoothing (media pesata)
        def smooth(slot, new, alpha=0.3):
            old = tracker.get_slot(slot)
            if old is None:
                return new
            return round((old * (1 - alpha)) + (new * alpha), 2)

        return [
            SlotSet("extraversion", smooth("extraversion", json_data.get("extraversion", 5))),
            SlotSet("emotional_stability", smooth("emotional_stability", json_data.get("emotional_stability", 5))),
            SlotSet("openness", smooth("openness", json_data.get("openness", 5))),
            SlotSet("agreeableness", smooth("agreeableness", json_data.get("agreeableness", 5))),
            SlotSet("conscientiousness", smooth("conscientiousness", json_data.get("conscientiousness", 5))),
        ]
    
    
class ActionSubmitIssueForm(Action):

    def name(self) -> str:
        return "action_submit_issue_form"
    
    def get_model(self) -> str:
        try:
            response = requests.get("http://localhost:1234/v1/models")
            if response.status_code == 200:
                models = response.json().get("data", [])
                if models:
                    return models[0]["id"]
        except Exception as e:
            print(f"Error: {e}")
        return "mistral-7b-instruct-v0.3"

    def build_prompt(self, current_issue: str, issue_data: Dict[str, Any]) -> str:
        issue_summary = "\n".join(
            [f"- {slot.replace('_', ' ').capitalize()}: {value}" for slot, value in issue_data.items()]
        ) if issue_data else "No further details were provided."

        return f"""You are an empathetic mental health support chatbot. The user has just completed a set of questions about their current 
        issue. The user don't know if the issue is true or not, it's just a predict, so avoid clinical labels or diagnoses.

            Here is what they've shared:

                - Current issue: {current_issue}
                                 {issue_summary}

            Please:
                1. Summarize the user’s concern in simple and compassionate terms.
                2. Suggest a gentle, preliminary interpretation of what might be happening 
                (avoid clinical labels or diagnoses, just possible patterns).
                3. Offer one or two supportive next steps the user might consider.

            Keep the tone warm, empathetic, non-judgmental, and supportive. 
            Do not introduce yourself, do not greet, and do not repeat your role — just respond naturally as if continuing an ongoing conversation.
            """

    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        current_issue = tracker.get_slot("current_issue")
        issues_profile = tracker.get_slot("issues_profile") or {}
        issue_data = issues_profile.get(current_issue, {})

        prompt = self.build_prompt(current_issue, issue_data)

        model = self.get_model()
        headers = { "Content-Type": "application/json" }
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        try:
            response = requests.post("http://localhost:1234/v1/chat/completions", json=payload, headers=headers)
            if response.status_code == 200:
                reply = response.json()['choices'][0]['message']['content']
            else:
                reply = "Thanks for completing the form! If you’d like, feel free to tell me if there’s anything you’d like to work on or explore together."
        except Exception as e:
            print(f"Error: {e}")
            reply = "Thanks for completing the form! If you’d like, feel free to tell me if there’s anything you’d like to work on or explore together."

        dispatcher.utter_message(text=reply)
        
        return []
    

