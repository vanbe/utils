#!/usr/bin/env bash

APP_NAME="Enregistreur Audio"
OUTPUT_DIR="$HOME"
OUTPUT_FILE=""
IS_RECORDING=false
IS_PAUSED=false
START_TIME=0
PAUSE_ACCUMULATED=0
PAUSE_START=0
REC_PID=0
AUDIO_DEVICE=""
AUDIO_BACKEND=""

# === Vérification dépendances ===
check_dependencies() {
    local deps=("sox" "zenity" "notify-send" "pactl")
    local missing=()
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" >/dev/null 2>&1; then
            missing+=("$dep")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        zenity --error --title="$APP_NAME" \
               --text="Dépendances manquantes : ${missing[*]}"
        exit 1
    fi
}

# === Choix de la source audio (PulseAudio) ===
choose_audio_device() {
    local sources=$(pactl list short sources | awk '{print $2}')
    local options=()
    local default_source=$(pactl info | grep "Default Source" | awk '{print $3}')

    for source in $sources; do
        if [[ "$source" == *"monitor"* ]]; then
            options+=("$source" "Sortie système (monitor)")
        elif [[ "$source" == *"usb"* ]] || [[ "$source" == *"bluez"* ]]; then
            options+=("$source" "Casque USB/Bluetooth")
        else
            options+=("$source" "Entrée micro")
        fi
    done

    AUDIO_DEVICE=$(zenity --list \
        --title="$APP_NAME" \
        --text="Choisissez la source d'enregistrement :" \
        --column="Source" --column="Type" \
        --print-column=1 \
        "${options[@]}")

    if [ -z "$AUDIO_DEVICE" ]; then
        exit 0
    fi
    AUDIO_BACKEND="pulseaudio"
}

# === Choix dossier ===
choose_output_dir() {
    OUTPUT_DIR=$(zenity --file-selection --directory \
        --title="Choisissez le dossier d'enregistrement")
    [ -z "$OUTPUT_DIR" ] && exit 0
}

# === Nouveau fichier ===
new_output_file() {
    local timestamp=$(date +'%Y%m%d-%H%M%S')
    OUTPUT_FILE="$OUTPUT_DIR/enregistrement-$timestamp.wav"
}

# === Démarrer enregistrement ===
start_recording() {
    new_output_file
    if sox --help 2>&1 | grep -q pulseaudio; then
        # sox sait gérer pulseaudio
        sox -t pulseaudio "$AUDIO_DEVICE" "$OUTPUT_FILE" &
    else
        # fallback sur arecord (via PulseAudio)
        arecord -D pulse -f cd "$OUTPUT_FILE" &
    fi

    REC_PID=$!
    sleep 0.5
    if ! kill -0 $REC_PID 2>/dev/null; then
        zenity --error --title="$APP_NAME" --text="Impossible de démarrer l'enregistrement."
        exit 1
    fi

    IS_RECORDING=true
    IS_PAUSED=false
    START_TIME=$(date +%s)
    PAUSE_ACCUMULATED=0
    notify-send "$APP_NAME" "Enregistrement démarré : $OUTPUT_FILE"
}


pause_recording() {
    if $IS_RECORDING && ! $IS_PAUSED; then
        kill -STOP "$REC_PID"
        IS_PAUSED=true
        PAUSE_START=$(date +%s)
        notify-send "$APP_NAME" "Enregistrement en pause."
    fi
}

resume_recording() {
    if $IS_RECORDING && $IS_PAUSED; then
        kill -CONT "$REC_PID"
        IS_PAUSED=false
        PAUSE_ACCUMULATED=$((PAUSE_ACCUMULATED + $(date +%s) - PAUSE_START))
        notify-send "$APP_NAME" "Enregistrement repris."
    fi
}

stop_recording() {
    if $IS_RECORDING; then
        kill -TERM "$REC_PID"
        wait "$REC_PID" 2>/dev/null
        IS_RECORDING=false
    fi
}

# === Durée ===
get_duration() {
    if $IS_RECORDING; then
        local now=$(date +%s)
        if $IS_PAUSED; then
            echo $((PAUSE_START - START_TIME - PAUSE_ACCUMULATED))
        else
            echo $((now - START_TIME - PAUSE_ACCUMULATED))
        fi
    else
        echo 0
    fi
}

# === Récapitulatif après arrêt ===
show_summary() {
    if [ ! -f "$OUTPUT_FILE" ]; then
        zenity --error --title="$APP_NAME" --text="Aucun fichier n'a été créé."
        return
    fi
    local duration=$(get_duration)
    local filesize=$(du -h "$OUTPUT_FILE" | awk '{print $1}')
    zenity --info --title="$APP_NAME" \
        --text="✅ Enregistrement terminé\n\nFichier : $OUTPUT_FILE\nDurée : ${duration}s\nTaille : $filesize"
}


# === Boucle GUI ===
main_loop() {
    # Étape 3 : écran principal
    choice=$(zenity --list --title="$APP_NAME" \
        --text="Prêt à enregistrer\n\nDossier : $OUTPUT_DIR\nSource : $AUDIO_DEVICE" \
        --column="Action" "Démarrer")
    [ -z "$choice" ] && exit 0

    if [ "$choice" = "Démarrer" ]; then
        start_recording
    fi

    # Étape 4 : enregistrement en cours
    while $IS_RECORDING; do
        local duration=$(get_duration)
        choice=$(zenity --list --title="$APP_NAME" \
            --text="Durée : ${duration}s\nSource : $AUDIO_DEVICE\nFichier : $(basename "$OUTPUT_FILE")" \
            --column="Action" "Pause" "Arrêter")
        if [ -z "$choice" ]; then
            # Bouton Annuler = quitter sans sauvegarder
            stop_recording
            rm -f "$OUTPUT_FILE"
            exit 0
        fi
        case $choice in
            "Pause")
                if $IS_PAUSED; then
                    resume_recording
                else
                    pause_recording
                fi
                ;;
            "Arrêter")
                stop_recording
                show_summary
                exit 0
                ;;
        esac
    done
}

### Exécution ###
check_dependencies
choose_audio_device
choose_output_dir
main_loop

