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
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        

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