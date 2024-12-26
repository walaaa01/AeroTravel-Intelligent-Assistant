import tkinter as tk
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained("fine_tuned_model")
model = DistilBertForSequenceClassification.from_pretrained("fine_tuned_model")

id_to_label = {
    0: "name", 
    1: "greeting", 
    2: "reserver billet", 
    3: "acceder reservation billet", 
    4:"max passengers", 
    5:"enfant",
    6:"mode paiement",
    7:"error carte credit", 
    8:"reserve CC",
    9:"enregistrement vol",
    10:"modification date",
    11:"changer nom",
    12:"annulation",
    13:"bagage en soute autorisee",
    14:"articles interdits en checked valises",
    15:"hand luggage",
    16:"articles interdits en cabine",
    17:"objets manquee de valise",
    18:"bagages dommage",
    19:"valises pas sur tapis",
}
category_responses = {
    "nom": ["je m'appele Fleebo.","Mon fournisseur ma nommé Fleebo :D","vous pouvez m'appeler Fleebo."],
    "salutation": ["Salut!","cc!","Hey! Comment je peux vous aider?","Salutation!"],
    "reserver billet" : ["Vous pouvez réserver jusqu'à 1h avant l'heure de départ ."],
    "acceder reservation billet" : ["Cliquez sur la rubrique 'Gérer ma réservation' et saisissez votre prénom, nom et numéro de dossier tel qu'ils sont motionnés sur le billet ."],
    "max passengers" : ["Neuf passagers, en incluant tous les adultes et les enfants de votre groupe."],
    "enfant" : ["Cette option n'est pas disponible en ligne, veuillez contacter le service client sur : +216 70 020 920 ou le +33187641000 ou l'un des points de vente de Nouvelair."],
    "mode paiement" : ["Nous acceptons que les cartes bancaires Visa ou Mastercard."],
    "error carte credit" : ["Si l'opération de paiement est refusée contactez le service client sur : +216 70 02 09 20 ou le +33 1 87 64 10 00 ou l'un des points de vente de Nouvelair."],
    "reserve CC" : ["Oui, vous pouvez le faire, mais il est essentiel de disposer de toutes les données personnelles du passager."],
    "enregistrement vol" : ["L'enregistrement commence 3 heures avant l'heure programmée de votre vol. L'heure limite de l'enregistrement est fixée à 60 minutes avant votre départ"],
    "modification date" : ["Cela est possible via  la rubrique 'Gérer ma réservation'."],
    "changer nom" : ["Le nom de famille n'est pas modifiable cependant vous pouvez modifier le prénom moyennant des frais en contactant le service client sur : +216 70 02 09 20 / +33 1 87 64 10 00 ou l'un des points de vente de Nouvelair."],
    "annulation" : ["Pour plus d'informations, veuillez consulter les conditions générales de vente via 'https://www.nouvelair.com/fr/contenu/vente', Vous pouvez demander l'annulation via ici : 'https://booking.nouvelair.com/ibe/feedback'"],
    "bagage en soute autorisee" : ["La franchise accordée dépend de votre destination. Pour plus de détails consultez ce lien : 'https://www.nouvelair.com/fr/bagage#Bagages_soute'"], 
    "articles interdits en checked valises" : ["Vous pouvez consulter la liste des articles non autorisés via ce lien : 'https://www.nouvelair.com/fr/contenu/articles-interdits-et-reglements'"],
    "hand luggage" : ["Pour toutes les destinations assurées par Nouvelair, un seul bagage en cabine est autorisé, ce bagage doit respecter les règles suivantes: -Poids maximum 8KG / -55 x 35 x 25 cm"], 
    "articles interdits en cabine" : ["Vous pouvez consulter la liste des articles non autorisés via ce lien : 'https://www.nouvelair.com/fr/contenu/articles-interdits-et-reglements'"],
    "objets manquee de valise" : ["Nous sommes désolés pour ce désagrément, veuillez déclarer l'incident au comptoir du service litige bagage situé à l'aéroport."],
    "bagages dommage" : ["Nous sommes désolés pour le désagrément, si vous n'avez pas encore rempli(s) le document PIR disponible au comptoir du service bagage à l’aéroport, veuillez accéder à ce lien : 'https://booking.nouvelair.com/ibe/feedback' retour vous sera assuré dans les brefs délais . NB : ce formulaire doit être rempli avant de quitter la zone douanière"],
    "valises pas sur tapis" : ["Nous sommes désolés que votre bagage n’ait pas été remis à l’arrivée. Avant de quitter l’aéroport veuillez déclarer l’incident au service litige bagage situé à l'aéroport. Si votre bagage n'a pas été localisé dans un délai de 21 jours veuillez remplir ce formulaire : 'https://booking.nouvelair.com/ibe/feedback'"],
}
def get_response(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id_to_label.get(predicted_class_id, "unknown")
    response = category_responses.get(predicted_label, ["Sorry, I don't understand that."])[0]
    return response

def send():
    user_input = entry.get()
    entry.delete(0, tk.END)
    response = get_response(user_input)
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "You: " + user_input + "\n")
    chat_log.insert(tk.END, "Bot: " + response + "\n\n\n")
    chat_log.config(state=tk.DISABLED)
    chat_log.yview(tk.END)

root = tk.Tk()
root.title("Nouvelair User Assistant")

chat_log = tk.Text(root, bd=0, bg="white", height="28", width="70", font="Arial", wrap=tk.WORD)
chat_log.config(state=tk.DISABLED)

scrollbar = tk.Scrollbar(root, command=chat_log.yview)
chat_log['yscrollcommand'] = scrollbar.set 

entry = tk.Entry(root, bd=0, bg="ivory", width="29", font="Arial")
entry.bind("<Return>", lambda event: send())

send_button = tk.Button(root, text="Send", command=send)

scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
chat_log.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
entry.pack(side=tk.LEFT, padx=5, pady=5)
send_button.pack(side=tk.LEFT)

root.mainloop()


