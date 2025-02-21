URL="https://archive.ics.uci.edu/static/public/94/spambase.zip"
DEST_DIR="data"
ZIP_FILE="spambase.zip"
echo "Downloading spambase.zip from $URL..."
curl -o $ZIP_FILE $URL
# Step 2: Create the 'data' directory if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
echo "Creating directory: $DEST_DIR"
mkdir $DEST_DIR
fi
echo "Extracting $ZIP_FILE..."
unzip $ZIP_FILE
if [ -f "spambase.data" ]; then
echo "Moving spambase.data to $DEST_DIR"
mv spambase.data $DEST_DIR/
else
exit 1
fi
echo "Cleaning up: Removing $ZIP_FILE"
rm $ZIP_FILE
echo "Download and extraction completed successfully."
rm spambase.names spambase.DOCUMENTATION
