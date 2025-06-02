from typing import Any, Optional, Text, Dict, List
from transformers import pipeline
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, ActiveLoop
from rasa_sdk.types import DomainDict
from rasa_sdk.forms import FormValidationAction
import requests

classifier = pipeline("zero-shot-classification",
                      model="Jiva/xlm-roberta-large-it-mnli", device=0, use_fast=True)              

candidate_labels = [
    "ansia", "stress", "alimentazione", "peso",
    "aderenza ai farmaci", "fumo", "alcol", 
    "motivazione", "dipendenze", "autostima", 
]



class ActionClassificaTopic(Action):

    def name(self) -> Text:
        return "action_classifica_topic"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text")

        topics = tracker.get_slot("topic_list") or []

     
        result = classifier(user_message, candidate_labels, multi_label=True)
        topic = result["labels"][0]
        score = result["scores"][0]

        
        if score > 0.8:
            dispatcher.utter_message(text=f"Ho capito che t'interessa discutere di {topic}.")
            
            if topic not in topics:
                updated_topics = topics + [topic]
            else: 
                updated_topics = topics

            return [
                SlotSet("topic_list", updated_topics),
                SlotSet("current_topic", topic)]
        else:
            dispatcher.utter_message(text="Non sono sicuro di quale sia l'argomento, puoi spiegarmi meglio?")
            return []





class ActionLLMFallback(Action):

    def name(self) -> Text:
        return "action_llm_fallback"
    
    def classify_topic(self, message: str) -> Optional[str]:
        result = classifier(message, candidate_labels, multi_label=True)
        if result['scores'][0] > 0.8:
            return result['labels'][0]
        return None
    
    def get_model(self) -> str:

        try:
            response = requests.get("http://localhost:1234/v1/models")
            if response.status_code == 200:
                models = response.json().get("data", [])
                if models:
                    return models[0]["id"] 
        except Exception as e:
            print(f"Errore nel recupero dei modelli: {e}")
        
        return "mistral-7b-instruct-v0.3"
     
    def run(self, 
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message.get('text')

        topic = self.classify_topic(user_message)
     
        if topic:
            current_list = tracker.get_slot("topic_list") or []
            if topic not in current_list:
                updated_topics = current_list + [topic]
                return [
                    SlotSet("current_topic", topic),
                    SlotSet("topic_list", updated_topics)
                ]
            else:
                return [SlotSet("current_topic", topic)]



        model_name = self.get_model()
        
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,  
            "messages": [
                {"role": "user", 
                 "content": user_message}
            ],
            "temperature": 0.5
        }

        response = requests.post("http://localhost:1234/v1/chat/completions", json=payload, headers=headers)
        
        if response.status_code == 200:
            reply = response.json()['choices'][0]['message']['content']
        else:
            reply = "Mi dispiace, si è verificato un errore."

        dispatcher.utter_message(text=reply)
        return []
    





class ActionImpostaConversazioneIniziata(Action):

    def name(self) -> Text:
        return "action_conversazione_avviata"
    
    def run(self, dispatcher, tracker, domain):
        return[SlotSet("conversazione_avviata", True)]
    

    



    
class ValidateFormInformazioniUtente(FormValidationAction):

    def name(self) -> Text:
        return "validate_form_informazioni_utente"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        

        if tracker.get_slot("form_completato") is True:
            return [ActiveLoop(None)]
        
     
        required_slots = ["nome", "eta", "luogo"]  

        filled_slots = [s for s in required_slots if tracker.get_slot(s) is not None]
        
        if len(filled_slots) == len(required_slots):           
            return [
                SlotSet("form_completato", True),
                ActiveLoop(None)
            ]

        return [] 




class ActionSubmitFormInformazioniUtente(Action):

    def name(self) -> str:
        return "action_submit_form_informazioni_utente"

    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: dict):
        
        nome = tracker.get_slot("nome")
        dispatcher.utter_message(text=f"Grazie {nome}! Ho salvato le tue informazioni. Raccontami un po' di come ti senti oggi.")

        
        return [
            SlotSet("form_completato", True),
            ActiveLoop(None)
        ]
    
