Delhi House Rent Predictor

Hello, this program is constructed with Machine Learning, which helps to find out the approximate honest price of house rent in Delhi (India) 
based on several parameters: from the coordinates of the house to the number of balconies.

---------------------

Step 1: Set Up (One-time only)
To run this, you need Python installed on your computer.

Open your code editor, it might be almost whatever (Pycharm, Visual Studio...), the most important in internet connection

Create a new file in your editor, name it predict.py and paste my code into it.

Then look for the Terminal and write there:
pip install numpy
pip install pandas
and so on, untill all the libraries ("import" lines in the code) will be downloaded.

---------------------

Step 2: Pick a Location
You can test specific neighborhoods using Google Maps:

Go to Google Maps and search for "Delhi".

Right-click on any street or building.

Click on the numbers (e.g., 28.527, 77.218). This copies the Latitude and Longitude.

---------------------

Step 3: Enter Your Data
In the script, look for the section marked #for user:. Change the numbers to whatever you like:

Variable	Description	Common Range (25th-75th Percentile)

latitude	South/North coordinate (from Google Maps)	28.457 to 28.603

longitude	West/East coordinate (from Google Maps)	77.138 to 77.228

numBathrooms	Number of bathrooms	2 to 4

numBalconies	Number of balconies	0 to 2

isNegotiable	Can you haggle? (0 = No, 1 = Yes)	0 (mostly non-negotiable)

verificationDate	Days since the ad was verified	20 to 365 days

Status 0 is unfurnished, 1 is semi-furnished, 2 is fully furnished	1 to 2

Size_m2	Total area of the real estate in square meters	120 to 548 m²

BHK	0 = Room/Kitchen, 1 = Bedroom/Hall/Kitchen	Usually 1

rooms_num	Total number of rooms	3 to 4

SecurityDeposit_euro	Upfront refundable deposit in Euros	0 to 10,829 €

---------------------

Step 4: Run the Script
Press the Run/Play button in your editor, and get the price

---------------------

Step 5: Reading the Results
The program will print several lines. Look at the very last one: RF prediction: XXX.XX

This is the estimated rent in Euro per month.

Note: I am using three different models (KNN, LR, and RF). RF (Random Forest) is the most accurate one, so please use that number for your feedback.

Feedback Request
After you try it a few times, please tell me:

Accuracy: Does the price change logically? (e.g., if you double the size, does the price go up reasonably?)

Ease of Use: Was it annoying to change the numbers inside the code?

Data: Were any of the input names confusing?
