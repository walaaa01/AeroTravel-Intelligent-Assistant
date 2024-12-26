from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import torch
'''import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('french'))

def eliminate_stopwords(text):
    tokens = text.split()
    filtered_tokens = []
    for word in tokens:
        if(word.lower() not in stop_words):
            filtered_tokens.append(word)
    return ' '.join(filtered_tokens)
'''
question_keywords = {
    
    "reserver billet": [
        "délai pour réserver", "délai minimum", "deadline réservation", 
        "dernier moment", "jusqu'à quand peut-on réserver ?", "quand dois-je réserver ?", 
        "date limite pour réservation", "avant quelle heure peut-on réserver ?", 
        "jusqu'à quand est-ce que je peux réserver ?", "quel est le délai pour réserver ?", 
        "peut-on réserver à la dernière minute ?", "quand est-ce que je dois réserver ?",
        "quelle est la date limite pour réserver ?", "est-ce que je peux réserver en avance ?", 
        "quel est le dernier délai pour la réservation ?", "y a-t-il une date limite pour réserver ?",
        "combien de temps avant le départ peut-on réserver ?","est-ce trop tard pour réserver ?" #18
    ],
    "acceder reservation billet": [
        "accéder à la réservation", "accéder à ma réservation", "consulter réservation", 
        "comment voir ma réservation ?", "comment accéder à ma réservation ?", "comment je peux acceder a mon reservation",
        "comment vérifier ma réservation ?", "comment consulter ma réservation ?", 
        "comment accéder à ma réservation en ligne ?", "où puis-je voir ma réservation ?",
        "comment accéder à ma réservation de vol ?", "où consulter les détails de ma réservation ?",
        "comment vérifier l'état de ma réservation ?", "comment accéder aux infos de ma réservation ?"
        "comment accéder aux détails de ma réservation ?", "comment voir les informations de ma réservation ?",
        "comment retrouver ma réservation ?", "où puis-je vérifier ma réservation ?" #18
    ],
    "max passengers": [
        "nombre passager", "nombre maximal", "nombre maximum des passagers",
        "combien de passagers", "quel est le nombre maximum de passagers ?", "quel est le nombre maximal des passagers",
        "combien de personnes maximum ?", "combien de passagers sont permis ?", "nombre maximal des passagers par réservation ?",
        "maximum de passagers par réservation ?", "quel est le nombre maximal de personnes autorisées ?",
        "combien de passagers peut-on ajouter ?", "quelle est la limite de passagers ?",
        "quel est le maximum de passagers autorisés ?", "combien de passagers puis-je inclure ?", 
        "combien de personnes au maximum par vol ?", "quelle est la capacité maximale de passagers ?",
        "quel est le nombre total de passagers permis ?" #18
    ],
    "enfant": [
        "enfant non accompagné", "enfant", "enfant voyageant tout seul", "mineur", 
        "mineur voyageant seul", "enfant seul", "je voyage avec un enfant seul", 
        "que faire avec un enfant non accompagné ?", "quels sont les services pour les mineurs ?",
        "enfant seul en voyage", "comment gérer un mineur seul ?", "je veux faire une reservation pour mon enfant",
        "enfant sans accompagnateur","procédures pour les mineurs voyageant seuls",
        "quels sont les services pour les enfants voyageant seuls ?", "comment gérer les mineurs voyageant seuls ?",
        "conditions pour enfants non accompagnés", "comment faire une reservation pour un enfant mineur qui va voyager tout seul" #18
    ],
    "mode paiement": [
        "méthodes de paiement", "modes de paiement", "comment je peux payer?",
        "payer", "carte de crédit", "carte bancaire", "est ce que vous acceptez les chéques?",
        "quels moyens de paiement sont acceptés ?", "quelles sont les options de paiement ?", 
        "comment puis-je payer ?", "acceptez-vous les chèques ?", "puis-je payer en espèces ?",
        "quels sont les moyens de paiement disponibles ?", "acceptez-vous PayPal ?",
        "puis-je payer par virement bancaire ?", "quels types de cartes de crédit sont acceptés ?", 
        "acceptez-vous les cartes de débit ?", "quelles méthodes de paiement acceptez-vous ?",
        "options de paiement disponibles", #19
    ],
    "error carte credit": [
        "refus de ma carte", "refus de mode de paiement", "carte de crédit a été refusée",
        "carte de crédit est refusée", "paiement échoué", "problème avec la carte de crédit",
        "problème avec la carte de crédit", "transaction refusée", "refus de paiement",
        "paiement par carte échoué", "refus de transaction bancaire", "carte bloquée",
        "carte non acceptée", "transaction échouée", "carte de crédit rejetée", "erreur de paiement", 
        "carte de crédit invalide", "erreur avec la carte de crédit", "erreur de transaction"
        "problème de paiement par carte", "problème de carte", "paiement rejeté" #21
       
    ],
    "reservation CC": [
        "réserver un billet avec ma carte de crédit pour une autre personne", 
        "autre personne", "quelqu'un d'autre", "réserver pour quelqu'un d'autre", 
        "réserver un billet pour quelqu'un d'autre", "peut-on réserver pour quelqu'un d'autre ?", 
        "est-il possible de payer pour une autre personne ?", "réserver en utilisant la carte d'une autre personne ?",
        "peut-on utiliser la carte de crédit d'un tiers ?", "réserver au nom d'une autre personne avec ma carte ?", 
        "payer pour quelqu'un d'autre avec ma carte ?", "réservation pour un tiers avec carte de crédit",
        "peut-on utiliser ma carte pour payer une réservation pour un ami ?", "est-il possible de payer pour un autre passager ?", 
        "puis-je faire une réservation pour quelqu'un d'autre avec ma carte de crédit ?", "faire une reservation pour un ami ou membre de famille avec ma carte de crédit",
        "réservation au nom d'un ami avec ma carte", "payer pour quelqu'un d'autre", "utiliser ma carte pour une autre personne" #19
    ],
    "enregistrement vol": [
        "enregistrement à l'aéroport", "enregistrement", "commence", "check-in",
        "quand commence l'enregistrement ?", "à quelle heure est l'enregistrement ?", 
        "quand puis-je m'enregistrer ?", "heure d'enregistrement", "quelles sont les heures d'enregistrement ?",
        "heures d'enregistrement pour mon vol", "à quel moment dois-je m'enregistrer ?", "quand puis-je commencer l'enregistrement ?",
        "quand est l'heure limite pour l'enregistrement ?", "heure d'ouverture du comptoir d'enregistrement", 
        "à quel moment l'enregistrement est-il disponible ?", "quelles sont les heures limites pour s'enregistrer ?",
        "quand est le début de l'enregistrement ?", "quelles sont les horaires pour le check-in ?", "quand est-ce que l'enregistrement se termine ?" #19
    ],
    "modification date": [
        "modifier la date", "changer la date", "reporter la date", "date de vol", 
        "date de mon vol", "peut-on changer la date du vol ?", "comment reporter une date ?", 
        "modifier la date de mon billet", "changer la date de départ", "peut-on modifier la date ?",
        "changement de date pour le vol", "reporter ou modifier la date de départ", "est-ce possible de changer la date ?",
        "comment faire pour changer la date de mon billet ?", "quelles sont les options pour modifier la date du vol ?", 
        "comment ajuster la date de mon voyage ?", "peut-on changer la date du vol après la réservation ?",
        "changement de date de réservation", "ajuster la date de vol", "modifier la date de départ" #20
    ],
    "changer nom": [
        "modifier mon nom", "nom", "modifier le nom", "prénom", "changer prénom",
        "comment changer mon nom dans ma réservation ?", "peut-on changer le nom sur le billet ?", 
        "modifier le nom sur la réservation", "changer le prénom", "modification du nom",
        "changer le nom du passager", "je veux changer mon nom", "comment mettre à jour mon nom sur la réservation ?", 
        "changer le nom sur le billet de vol", "comment mettre à jour le nom sur ma réservation ?", 
        "quelles sont les procédures pour changer le nom sur le billet ?", "peut-on modifier le nom après réservation ?",
        "comment corriger un nom sur une réservation ?", "changement de nom sur le billet", "mettre à jour mon prénom sur le billet" #20
    ],
    "annulation": [
        "annuler la réservation", "annuler ma réservation", "annuler une réservation", 
        "annulation de ma réservation", "annulation d'une réservation", "je veux annuler mon voyage, comment faire ca",
        "comment annuler une réservation ?", "quelle est la procédure d'annulation ?", 
        "comment obtenir un remboursement ?", "peut-on annuler une réservation ?","comment obtenir le remboursement après annulation ?",
        "annulation de billet de vol", "que faire pour annuler une réservation ?", "procédure pour annuler une réservation",
        "quels sont les frais d'annulation ?", "comment faire une demande d'annulation ?", 
        "comment annuler une réservation de vol ?", "quelles sont les conditions d'annulation ?",
        "comment procéder pour annuler mon billet ?", "annuler réservation en ligne" #20
    ],
    "bagage en soute autorisee": [
        "bagage en soute autorisé", "limite poids soute", "valise en soute", "poids maximal pour les bagages en soute"
        "quels sont les poids autorisés en soute ?", "poids maximum pour les bagages en soute", 
        "limite de poids pour la valise en soute", "combien de kilos pour les bagages en soute ?", 
        "quelles sont les restrictions de poids pour les bagages en soute ?", "limites de poids pour les bagages",
        "quel est le poids autorisé pour les valises en soute ?", "combien de poids puis-je mettre en soute ?",
        "quelle est la limite de poids pour les bagages en soute ?", "quels sont les poids autorisés pour les bagages enregistrés ?", 
        "quel est le maximum de poids pour les bagages en soute ?", "limites de poids pour les bagages enregistrés",
        "poids autorisé pour bagage enregistré", "combien de kilos autorisés en soute ?", "limitations de poids pour valises en soute" #19
    ],
    "articles interdits en checked valises": [
        "articles interdits en soute", "objets interdits en soute", "produits interdits en soute", 
        "bagages interdits en soute", "non autorisé en soute", "quels articles ne sont pas permis en soute ?", 
        "liste des articles interdits en soute", "que ne puis-je pas mettre dans ma valise en soute ?", 
        "articles à ne pas mettre en soute", "quoi ne pas inclure dans les bagages en soute ?",
        "quels objets sont interdits dans les bagages en soute ?", "articles à éviter dans les valises en soute",
        "quels articles sont proscrits pour les bagages en soute ?", "que ne puis-je pas mettre dans les bagages enregistrés ?", 
        "interdictions pour les bagages en soute", "objets non autorisés dans les bagages enregistrés",
        "articles interdits en valise", "liste des articles interdits pour bagages en soute", "objets non permis en soute" #19
    ],
    "hand luggage": [
        "poids maximal en cabine", "règles de bagages à main", "poids maximum en cabine", "bagage à main",
        "quelles sont les règles pour le bagage à main ?","poids maximal autorisé pour les bagages à main", "poids maximal autorisé pour les bagages en cabine",
        "dimensions maximales du bagage à main", "limites de poids pour le bagage cabine", "poids maximal a main",
        "règles sur le bagage de cabine", "combien de kilos pour le bagage à main ?", "quel est le poids maximal pour bagage en cabine",
        "quelles sont les dimensions autorisées pour les bagages à main ?","poids et dimensions du bagage à main"
        "règles pour les bagages de cabine", "quelles sont les restrictions pour les bagages à main ?", 
        "combien de bagages à main puis-je emporter ?","combien de bagages en cabine puis-je emporter ?",
        "dimensions du bagage à main", "quels sont les règles pour les bagages en cabine" #21
    ],
    "articles interdits en cabine": [
        "articles interdits en cabine", "objets interdits en cabine", "produits interdits en cabine", 
        "bagages interdits en cabine", "non autorisé en cabine", "quels objets ne sont pas permis en cabine ?", 
        "liste des articles interdits en cabine", "que ne puis-je pas mettre dans le bagage à main ?", 
        "articles à ne pas mettre en cabine", "quoi ne pas inclure dans le bagage cabine ?", 
        "objets non autorisés en cabine", "liste des articles interdits pour le bagage à main",
        "quels articles sont interdits dans les bagages de cabine ?", "articles prohibés en cabine",
        "quels sont les articles à éviter en cabine ?", "liste des objets non permis en cabine",
        "articles interdits à bord", "que dois-je éviter d'emporter en cabine ?", "objets non permis en cabine" #19
    ],
    "objets manquées de valise": [
        "manque certains objets", "objets manqués", "objets manquants","objets manquants dans ma valise", 
        "valise volée", "objet volé", "manque des articles dans ma valise", 
        "que faire lorsque des objets manquent dans mon bagage ?", "que faire lorsqu'il y a des objets manqués dans mon bagage ?", 
        "que faire lorsqu'il y a des objets manqués dans mes valises ?", "que faire si des articles sont absents de ma valise ?", 
        "mon bagage est incomplet, que faire ?", "comment signaler des objets manquants dans ma valise ?", 
        "objets perdus dans la valise", "comment faire une déclaration pour objets manquants ?", 
        "que faire en cas de perte d'objets dans la valise ?", "mon bagage est volée, comment signaler ?",
        "bagage incomplet", "perte de bagage", "valise ouverte" #22
    ],
    "bagages dommage": [
        "bagages endommagés", "bagages subissent un dommage", "valise endommagée", "comment signaler des dommages sur mon bagage ?",
        "que faire si mon bagage est endommagé ?", "qu'est ce que je fait lorsque dommages sur mes valises",  
        "bagage abîmé", "dommages sur la valise", "valise endommagée, comment signaler ?", "mon bagage a été endommagé, que faire ?", 
        "bagages endommagés, quelle procédure suivre ?", "que faire si mes valises ont été endommagés ?", 
        "comment déclarer des dommages sur mes bagages ?", "comment obtenir une compensation pour bagages endommagés ?", 
        "que faire en cas de dommage sur mon bagage ?", "bagage cassé", "que faire si mes valises sont endommagées ?",
        "problème de bagage", "dommage de valise", "bagage défectueux", "valise endommagée" #21
    ],
    "valises pas sur tapis": [
        "tapis", "valises perdues", "bagages pas sur le tapis", "où est ma valise ?",
        "valise non trouvée", "bagage disparu", "valise non sur le tapis", "problème de tapis à bagages",
        "mon bagage n'est pas sur le tapis de réception", "que faire si ma valise n'est pas sur le tapis ?", 
        "où récupérer ma valise si elle n'est pas sur le tapis ?", "qu'est-ce que je dois faire lorsque mes valises sont perdues",
        "qu'est-ce que je dois faire lorsque je ne trouve pas mon bagage sur le tapis", "comment signaler perte de mes valises",
        "valise manquante","où est mon bagage", "ou sont mes valises", "que faire lorsque mon bagage n'est pas sur le tapis",
        "perte de valise", "valise n'a pas arrivée", "comment signaler perte de mon bagage" #22
    ]
}

category_responses = {
    
    "reserver billet" : ["Vous pouvez réserver jusqu'à 1h avant l'heure de départ ."],
    "acceder reservation billet" : ["Cliquez sur la rubrique 'Gérer ma réservation' et saisissez votre prénom, nom et numéro de dossier tel qu'ils sont motionnés sur le billet ."],
    "max passengers" : ["Neuf passagers, en incluant tous les adultes et les enfants de votre groupe."],
    "enfant" : ["Cette option n'est pas disponible en ligne, veuillez contacter le service client sur : +216 70 020 920 ou le +33187641000 ou l'un des points de vente de Nouvelair."],
    "mode paiement" : ["Nous acceptons que les cartes bancaires Visa ou Mastercard."],
    "error carte credit" : ["Si l'opération de paiement est refusée contactez le service client sur : +216 70 02 09 20 ou le +33 1 87 64 10 00 ou l'un des points de vente de Nouvelair."],
    "reservation CC" : ["Oui, vous pouvez le faire, mais il est essentiel de disposer de toutes les données personnelles du passager."],
    "enregistrement vol" : ["L'enregistrement commence 3 heures avant l'heure programmée de votre vol. L'heure limite de l'enregistrement est fixée à 60 minutes avant votre départ"],
    "modification date" : ["Cela est possible via  la rubrique 'Gérer ma réservation'."],
    "changer nom" : ["Le nom de famille n'est pas modifiable cependant vous pouvez modifier le prénom moyennant des frais en contactant le service client sur : +216 70 02 09 20 / +33 1 87 64 10 00 ou l'un des points de vente de Nouvelair."],
    "annulation" : ["Pour plus d'informations, veuillez consulter les conditions générales de vente via 'https://www.nouvelair.com/fr/contenu/vente', Vous pouvez demander l'annulation via ici : 'https://booking.nouvelair.com/ibe/feedback'"],
    "bagage en soute autorisee" : ["La franchise accordée dépend de votre destination. Pour plus de détails consultez ce lien : 'https://www.nouvelair.com/fr/bagage#Bagages_soute'"], 
    "articles interdits en checked valises" : ["Vous pouvez consulter la liste des articles non autorisés via ce lien : 'https://www.nouvelair.com/fr/contenu/articles-interdits-et-reglements'"],
    "hand luggage" : ["Pour toutes les destinations assurées par Nouvelair, un seul bagage en cabine est autorisé, ce bagage doit respecter les règles suivantes: -Poids maximum 8KG / -55 x 35 x 25 cm"], 
    "articles interdits en cabine" : ["Vous pouvez consulter la liste des articles non autorisés via ce lien : 'https://www.nouvelair.com/fr/contenu/articles-interdits-et-reglements'"],
    "objets manquées de valise" : ["Nous sommes désolés pour ce désagrément, veuillez déclarer l'incident au comptoir du service litige bagage situé à l'aéroport."],
    "bagages dommage" : ["Nous sommes désolés pour le désagrément, si vous n'avez pas encore rempli(s) le document PIR disponible au comptoir du service bagage à l’aéroport, veuillez accéder à ce lien : 'https://booking.nouvelair.com/ibe/feedback' retour vous sera assuré dans les brefs délais . NB : ce formulaire doit être rempli avant de quitter la zone douanière"],
    "valises pas sur tapis" : ["Nous sommes désolés que votre bagage n’ait pas été remis à l’arrivée. Avant de quitter l’aéroport veuillez déclarer l’incident au service litige bagage situé à l'aéroport. Si votre bagage n'a pas été localisé dans un délai de 21 jours veuillez remplir ce formulaire : 'https://booking.nouvelair.com/ibe/feedback'"],

}

def prepare_dataset(question_keywords):
    inputs= []
    labels = []
    for category, keywords in question_keywords.items():
        for keyword in keywords:
            inputs.append(keyword)
            labels.append(category)
    return Dataset.from_dict({"text": inputs, "label": labels}) 

dataset = prepare_dataset(question_keywords)

labels = list(question_keywords.keys())
label_to_id = {}
id_to_label = {}
for i, label in enumerate(labels):
    label_to_id[label] = i 
for i, label in enumerate(labels):
    id_to_label[i] = label 


def label_encoder(text):
    text['label'] = label_to_id[text['label']]
    return text

dataset = dataset.map(label_encoder)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(labels))

def tokenize_function(text):
    return tokenizer(text["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

metric = load_metric("accuracy", trust_remote_code=True)

def compute_metrics(p):
    logits, labels = p
    logits_tensor = torch.tensor(logits) 
    predictions = torch.argmax(logits_tensor, dim=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16, 
    num_train_epochs=6, 
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
