import concurrent.futures
import subprocess

# Liste d'adresses URL à vérifier
url_list = [
    "https://pypi.org",
    "https://github.com",
    "https://www.google.com",
    "https://www.wikipedia.org",
    "https://www.microsoft.com",
    "https://repo.continuum.io",
    "https://conda.anaconda.org",
    "https://vscode-update.azurewebsites.net/",
    "https://repo.anaconda.com",
    "https://github.com",
    "https://huggingface.co",
    "https://cdn-lfs.huggingface.co",
    "https://hf.co/",
    "https://pypi.python.org",
    "https://pypi.org",
    "https://pythonhosted.org",
    "https://files.pythonhosted.org",
    "https://download.pytorch.org"


    # Ajoutez autant d'URL que vous voulez vérifier
]

def check_url(url):
    """
    Vérifie si les flux sont ouverts pour une adresse URL donnée à l'aide de la commande curl.
    """
    try:
        # Exécute la commande curl dans un sous-processus
        output = subprocess.check_output(["curl", "--head", "--silent", url], universal_newlines=True)
        # Vérifie si le code de réponse HTTP est 200 (OK)
        if "HTTP/1.1 200 OK" in output:
            print(f"{url} : flux ouvert")
        else:
            print(f"{url} : flux fermé ou erreur")
    except subprocess.CalledProcessError as e:
        print(f"{url} : erreur de connexion ({e.returncode})")

# Crée une ThreadPoolExecutor avec 5 threads de travail
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Soumet les tâches pour vérifier les URL en parallèle
    futures = [executor.submit(check_url, url) for url in url_list]
    # Attend que toutes les tâches soient terminées
    concurrent.futures.wait(futures)