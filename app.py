import ghhops_server as hs
from flask import Flask
import rhino3dm as rh
import joblib

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import numpy as np


loaded_model = joblib.load('gradient_boosting_model.joblib')
app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)


# Define the Objective functions
# Objective Functions from https://machinelearningmastery.com/curve-fitting-with-python/
def objective(x, a, b):
    return a * x + b

def objective2(x, a, b, c):
    return a * x + b * x**2 + c

def objective3(x, a, b, c, d, e, f):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

def fit_to_objective(xdata,ydata,obj_fun = 0):

    if obj_fun==0:
        parameters, _ = curve_fit(objective, xdata, ydata)
        a,b = parameters
        x_line = np.arange(min(xdata), max(xdata), 1)
        y_line = objective(x_line, a, b)
    elif obj_fun==1:
        parameters, _ = curve_fit(objective2, xdata, ydata)
        a,b,c = parameters
        x_line = np.arange(min(xdata), max(xdata), 1)
        y_line = objective2(x_line, a, b, c)
    elif obj_fun==2:
        parameters, _ = curve_fit(objective3, xdata, ydata)
        a, b, c, d, e, f = parameters
        x_line = np.arange(min(xdata), max(xdata), 1)
        y_line = objective3(x_line, a, b, c,d,e,f)
    else:
        x_line,y_line = [],[]
    return x_line,y_line

@app.route("/index")
def index():
    return ("This is my Flask app")

@hops.component(
    "/sum",
    name="Sum",
    description="sum of numbers",
    inputs=[
        hs.HopsNumber("A", "A", "First Number"),
        hs.HopsNumber("B", "B", "Second Number")
    ],
    outputs=[
        hs.HopsNumber("Sum", "Sum", "Sum of the numbers")
    ]
)
def sum (a:float, b:float):
    return a+b

@hops.component(
    "/plotFit",
    name = "Plt",
    description = "This component plot the point and save the png files",
    inputs = [hs.HopsBoolean("Plot","Plot","True to Plot",access=hs.HopsParamAccess.ITEM),
              hs.HopsPoint("Points","Points","Points to plot",access=hs.HopsParamAccess.LIST),
              hs.HopsCurve("Curve","Curve","Fit curve to plot",access=hs.HopsParamAccess.ITEM),
              hs.HopsString("File Name","File Name","File Name",access=hs.HopsParamAccess.ITEM)
    ],
    outputs=[hs.HopsBoolean("Success","Success","True if plotted False if error",access=hs.HopsParamAccess.ITEM)
    ]
)
def plotFit(save,pts,crv,file_name):
    X = [pt.X for pt in pts]
    Y = [pt.Y for pt in pts]

    crv.Domain = rh.Interval(0.00,1.00)
    cX = []
    cY = []
    for i in range(len(pts)):
        pt = crv.PointAt(i/len(pts))
        cX.append(pt.X)
        cY.append(pt.Y)
    if save:
        fig,ax = plt.subplots()
        ax.plot(X, Y, 'o', label='data')
        ax.plot(cX, cY, '-', label='fit')
        ax.legend()
        name = "{}.png".format(file_name)
        fig.savefig(name)
        return True
    
    return False

@hops.component(
    "/fitCurve",
    name = "fit",
    description = "This component fit a curve to a set of points",
    inputs = [hs.HopsPoint("Points","Points","Points to plot",access=hs.HopsParamAccess.LIST),
              hs.HopsNumber("Objective","Obj","Objective Type",access=hs.HopsParamAccess.ITEM)
    ],
    outputs=[hs.HopsBoolean("Success","Success","True if fitted curve",access=hs.HopsParamAccess.ITEM),
            hs.HopsCurve("fitCurve","CRV","Fitted curve",access=hs.HopsParamAccess.ITEM)
    ]
)
def fitCurve(pts,obj):
    X = [pt.X for pt in pts]
    Y = [pt.Y for pt in pts]
    Z = [pt.Z for pt in pts]

    xdata = np.asarray(X)
    ydata = np.asarray(Y)

    x_line,y_line = fit_to_objective(xdata,ydata,obj_fun=obj)

    newPoints = []
    if len(x_line)>0:
        for i in range(len(x_line)):
            pt = rh.Point3d(x_line[i],y_line[i],0)
            newPoints.append(pt)
        crv = rh.Curve.CreateControlPointCurve(newPoints,1)
        print ("I'm here now")
        return True,crv
    print ("lol")
    crv = rh.Curve.CreateControlPointCurve(pts,1)
    print (crv)
    return False,crv

@hops.component(
    "/savePoints",
    name = "savePoints",
    description = "This component fit a curve to a set of points",
    inputs = [hs.HopsPoint("Points","Points","Points to plot",access=hs.HopsParamAccess.LIST)
    ],
    outputs=[hs.HopsBoolean("Success","Success","True if plotted False if error",access=hs.HopsParamAccess.ITEM)
    ]
)
def savePoints(pts):
    X = [pt.X for pt in pts]
    Y = [pt.Y for pt in pts]
    Z = [pt.Z for pt in pts]
    
    points_dict = {'X':X,'Y':Y,'Z':Z}
    
    file_path = 'points.json'


    with open(file_path, 'w') as json_file:
        json.dump(points_dict, json_file, indent=4)

    print(f"JSON file saved at: {file_path}")
        
    return True

@hops.component(
    "/predictPoint",
    name = "predictPoint",
    description = "This component fit a curve to a set of points",
    inputs = [hs.HopsPoint("Points","Points","Points to plot",access=hs.HopsParamAccess.ITEM)
    ],
    outputs=[hs.HopsPoint("Points","Points","Points to plot",access=hs.HopsParamAccess.ITEM)
    ]
)
def predictPoint(pt):
    X = pt.X 
    Y = pt.Y
    print (X,Y)
    # Load the saved model

    print ("lol")
    new_data = np.array([[X,Y]])
    prediction = loaded_model.predict(new_data)

    new_pt = rh.Point3d(X,Y,prediction)
    return new_pt


if __name__ == "__main__":

    app.run(debug=True)



