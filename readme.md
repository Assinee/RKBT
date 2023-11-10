git clone https://github.com/ultralytics/yolov5
pip install -r requirements.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1OYwrlRti4cieuvVr8ERaJhTQdFJXWT4I' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1OYwrlRti4cieuvVr8ERaJhTQdFJXWT4I" -O best.pt && rm -rf /tmp/cookies.txt

for window
# Define the URL and the destination file name
$url = "https://docs.google.com/uc?export=download&id=1OYwrlRti4cieuvVr8ERaJhTQdFJXWT4I"
$outputFile = "best.pt"

# Use PowerShell to fetch the confirmation token
$confirmation = (Invoke-WebRequest -Uri $url -UseBasicParsing).Content -match "confirm=([0-9A-Za-z_]+)"
$confirmationToken = $matches[1]

# Use the confirmation token to download the file
Invoke-WebRequest -Uri ("https://docs.google.com/uc?export=download&confirm=" + $confirmationToken + "&id=1OYwrlRti4cieuvVr8ERaJhTQdFJXWT4I") -OutFile $outputFile

python detect.py --weights ../best.pt --img 1280 --conf 0.25 --source ../clips/08fd33_4.mp4 --name custom

!git clone https://github.com/ifzhang/ByteTrack.git
!cd ByteTrack && pip3 install -r requirements.txt
!cd ByteTrack && python3 setup.py develop
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox
pip install moviepy

!pip install onemetric --quiet
!pip install loguru
!pip install lap