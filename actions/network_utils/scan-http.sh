#!/bin/bash

# Vérification des dépendances
if ! command -v nmap &> /dev/null; then
    echo "nmap n'est pas installé. Installation en cours..."
    sudo apt update && sudo apt install -y nmap
fi

if ! command -v curl &> /dev/null; then
    echo "curl n'est pas installé. Installation en cours..."
    sudo apt install -y curl
fi

# Récupération de la plage réseau
network=$(ip route | awk '/default/ {split($3, ip, "."); print ip[1]"."ip[2]"."ip[3]".0/24"}')
echo "Scan du réseau: $network"

# Scan Nmap des IP avec port 88 ouvert
echo "Recherche des hôtes avec le port 88 ouvert..."
nmap -p 88 --open -oG - $network | awk '/Up$/{print $2}' > ips.txt

# Vérification HTTP
echo "Vérification des services HTTP sur le port 88..."
echo "---------------------------------------------"
found=0
while read ip; do
    echo "Testing $ip:88"
    response=$(curl -Is --connect-timeout 3 "http://$ip:88" | head -n 1)
    
    if [[ "$response" == *"HTTP/"* ]]; then
        echo "✅ Service HTTP actif détecté!"
        echo "$ip:88 - $response"
        echo "---------------------------------------------"
        found=1
    else
        echo "❌ Aucun service HTTP détecté"
        echo "---------------------------------------------"
    fi
done < ips.txt

# Nettoyage et résultat
rm ips.txt
if [ "$found" -eq 0 ]; then
    echo "Aucun service HTTP valide trouvé sur le port 88"
fi
