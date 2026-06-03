#!/bin/bash
# Lance l'UI web de revue des near-duplicates sur dockerlocal.
# À exécuter SUR dockerlocal (docker + image utils:latest + NAS CIFS /mnt/nas-homes).
# Accès navigateur : http://192.168.1.15:8081
#
#   bash review.sh                 # candidats par défaut : Drive/.perceptual_candidates.json
#   bash review.sh --batch 80      # passe des options à review_server.py
#
# Ctrl+C pour arrêter (le conteneur est --rm).
exec docker run --rm -p 8081:8081 \
  -v /root/code/utils:/opt/utils:ro \
  -v /mnt/nas-homes:/nas \
  utils:latest \
  python /opt/utils/actions/picture_utils/review_server.py "$@"
