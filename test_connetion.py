import os
import sys

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configuration des couleurs pour un rendu propre dans la console
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(title):
    print(f"\n{BLUE}=== {title} ==={RESET}")

def test_openai_connection():
    print_header("Test de connexion : OpenAI API")
    
    if not OPENAI_AVAILABLE:
        print(f"{RED}❌ Erreur : La bibliothèque 'openai' n'est pas installée dans cet environnement.{RESET}")
        print("👉 Lancez : pip install -r requirements.txt")
        return False
        
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"{YELLOW}⚠️ Attention : La variable d'environnement OPENAI_API_KEY n'est pas configurée !{RESET}")
        print("👉 Veuillez la configurer avant d'exécuter ce script ou la saisir directement dans l'interface de l'application.")
        return False
        
    print(f"🔑 Clé API trouvée (commençant par : {api_key[:8]}...)")
    
    try:
        print("📡 Envoi d'une requête de ping sémantique à api.openai.com...")
        client = OpenAI()
        # Appel d'un modèle léger pour limiter la consommation de jetons
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Dit 'Connexion OK' en 2 mots."}],
            max_tokens=10,
            temperature=0
        )
        answer = response.choices[0].message.content.strip()
        print(f"{GREEN}✅ Connexion OpenAI réussie !{RESET}")
        print(f"💬 Réponse de gpt-4o-mini : \"{answer}\"")
        return True
    except Exception as e:
        print(f"{RED}❌ Échec de la connexion à OpenAI : {e}{RESET}")
        print("\n💡 Suggestions de dépannage :")
        print("1. Vérifiez que votre clé API est valide, active et qu'elle dispose de fonds suffisants.")
        print("2. Si vous êtes derrière un proxy d'entreprise, assurez-vous de l'avoir configuré dans votre terminal.")
        return False

if __name__ == "__main__":
    print(f"{BLUE}===============================================")
    print("🔬 DIAGNOSTIC DE CONNEXION : MEDALLION STUDIO (CLOUD)")
    print(f"==============================================={RESET}")
    
    openai_ok = test_openai_connection()
    
    print_header("Bilan de configuration du système")
    if openai_ok:
        print(f"{GREEN}● OpenAI : PRÊT (Toutes fonctionnalités opérationnelles !){RESET}")
    else:
        print(f"{YELLOW}● OpenAI : NON CONFIGURÉ (L'application Streamlit vous permettra de saisir votre clé en direct.){RESET}")
    print(f"{BLUE}==============================================={RESET}\n")