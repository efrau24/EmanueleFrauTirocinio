from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, ActiveLoop
from rasa_sdk.types import DomainDict
from rasa_sdk.forms import FormValidationAction
import requests


class ActionLLMFallback(Action):

    def name(self) -> Text:
        return "action_llm_fallback"
     
    def run(self, 
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message.get('text')
        
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistral-7b-instruct-v0.3",  
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
    


class ActionAggiornaListaTopic(Action):

    
    def name(self) -> str:
        return "action_aggiorna_lista_topic"
    
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        topics = tracker.get_slot("topic_list") or []
        new_topic = next(tracker.get_latest_entity_values("topic"), None)
        
        if new_topic:
                
            if new_topic not in topics:
                topics.append(new_topic)
                updated_topics = topics
            else:
                updated_topics = topics

        else:
            updated_topics = topics
            


        return[SlotSet("topic_list", updated_topics)]