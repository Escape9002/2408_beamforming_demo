#! /bin/bash
# set -e
############ HOW TO RUN ####################
# run this script WITHOUT sudo and give    #
# your git acc and login data, otherwise   #
# the repository can't be setup for you!   #
############################################

############### SCRIPT PARAMETERS

current_root=$(pwd)
VENV_NAME=beam11_venv

FOLDER_EXISTS=$(test -d "$VENV_NAME" && echo true || echo false)

if ! $FOLDER_EXISTS ; then
    echo '>>>>>>>>>>>>>> NO FOLDER'
    
    echo 'Someone forgot to init this Laptop.
    If you happen to have internet run the init.sh file,
    If you dont have Internet, copy the UARM_env from another laptop into this directory.
    If you have neither...go buy some donuts, you will need them! '
    exit 1
fi

# Define file paths
FILES=(
    "src/config/config.py"
    "src/config/config_sac_learner.py"
    "src/config/config_error_model.py"
)

# Loop through each file
for FILE in "${FILES[@]}"; do
    if [ -f "$FILE" ]; then
        echo "$FILE exists. Skipping."
    else
        # If file does not exist, create it from its .default version
        DEFAULT_FILE="${FILE}.default"
        if [ -f "$DEFAULT_FILE" ]; then
            echo "$FILE does not exist. Creating it from $DEFAULT_FILE."
            cp "$DEFAULT_FILE" "$FILE"
        else
            echo "Error: Default file $DEFAULT_FILE does not exist. Cannot create $FILE."
        fi
    fi
done

############### START JUPYTER PROJECT/SERVER

source $VENV_NAME/bin/activate

python src/gui.py
