from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore, auth
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, Band
from bokeh.embed import components
from bokeh.palettes import Spectral11
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.layouts import row, column
from sklearn.model_selection import KFold, ShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import gaussian_kde
from sklearn.svm import SVR
import joblib
import os
import json

cred = credentials.Certificate("C:\\Users\\Acer\\Desktop\\machine_learning\\test\\ml\\ServiceAccountKey.json")
app = firebase_admin.initialize_app(cred)

config = {
    "apiKey": "AIzaSyC-YKbup4vKevTPAOHFzv_gLrjoYrJrCI8",
    "authDomain": "modelling-acadf.firebaseapp.com",
    "databaseURL": "https://modelling-acadf-default-rtdb.firebaseio.com",
    "projectId": "modelling-acadf",
    "storageBucket": "modelling-acadf.appspot.com",
    "messagingSenderId": "30422085238",
    "appId": "1:30422085238:web:5ec68ea3b2a78f92633957",
    "measurementId": "G-90M5PT2TQJ"
    }

firebase = pyrebase.initialize_app(config)
firebase_auth = firebase.auth()


def newuser(request):
    return render (request,"signup.html")

def postSignUp(request):
    email = request.POST.get('email')
    password = request.POST.get('password')
    try:
        firebase_auth.create_user_with_email_and_password(email, password)
        success = "Registration Successful. You can now login"
        return render(request, "signIn.html",{"success":success})
    except:
        message = 'Invalid Credentials. Please Try Again'
        print(message)
        return render(request, "signup.html", {"message": message})

def signIn(request):
    return render (request,"signIn.html")

def postSignIn(request):
    email = request.POST.get('email')
    password = request.POST.get('password')

    try:
        user = firebase_auth.sign_in_with_email_and_password(email, password)
        return render(request, 'home_class.html')
    except:
        message = 'Invalid Credentials. Please Try Again'
        return render(request, "signIn.html", {"message": message})
    
def log_out(request):
    try:
        firebase_auth.current_user = None  # Sign out the user from Firebase
    except KeyError:
        pass
    return redirect('/')

def home_class (request):
    return render (request, "home_class.html")

def firestore_data_view(request):
    store = firestore.client()
    doc_ref = store.collection(u'rice')

    df = []

    docs = doc_ref.get()
    for doc in docs:
        df.append(doc.to_dict())


    # Convert the list of dictionaries into a Pandas DataFrame
    data = pd.DataFrame(df)
    data_head_json = data.to_json(orient='split')

    #Create a Bar Graph
    class_frequency = data['Class'].value_counts()
    class_names = class_frequency.index.tolist()
    frequencies = class_frequency.values.tolist()
    
    source = ColumnDataSource(data=dict(Class=class_names, Frequency=frequencies))

    # Create a Bokeh bar figure
    bar = figure(x_range=class_names, height=350, width=450, background_fill_color='rgba(0, 0, 0, 0)', 
           toolbar_location=None, tools="")
    
    # Add bars to the figure
    bar.vbar(x='Class', top='Frequency', width=0.9, source=source)

    # Create a HoverTool
    hover = HoverTool(
        tooltips=[
            ('Class', '@Class'),
            ('Frequency', '@Frequency'),
        ],
        mode='vline' 
    )

    # Add the HoverTool to the figure
    bar.add_tools(hover)

    # Specify two colors for the bars
    colors = ['#004F7C','#A2346F',]

    # Map factors to colors using factor_cmap
    color_map = factor_cmap('Class', palette=colors, factors=class_names)

    # Add bars to the figure with the specified colors
    bar.vbar(x='Class', top='Frequency', width=0.9, color=color_map, source=source)

    # Customize the plot (optional)
    bar.title.text_font_size = '16pt'
    bar.axis.major_label_text_color = '#fff'
    bar.xaxis.major_label_text_font_size = '14pt'
    bar.yaxis.major_label_text_font_size = '14pt'
    bar.border_fill_color = '#00000000' 
    bar.xgrid.grid_line_color = None
    bar.ygrid.grid_line_color = None

   # Convert numeric columns to numeric type
    numeric_columns = ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 'Eccentricity', 'Convex_Area', 'Extent']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Create Bokeh components for bar chart
    script_bar, div_bar = components(bar)

    # Loop through each numeric column and create a density plot
    density_plots = []
    original_cmap = ['#A2346F', '#8563AE', '#e31a1c', '#8D7257', '#374955', '#007260', '#004F7C']

    for numeric_column, color in zip(numeric_columns, original_cmap):
        density_plot = figure(title=f'{numeric_column} Density Plot', height=300, width=335, background_fill_color="rgba(0, 0, 0, 0)", margin=(10, 10, 10, 10), toolbar_location=None, tools="hover")
        density_plot.title.text_color = "#fff"

        for class_name in class_names:
            density_data = data[data['Class'] == class_name][numeric_column]

            # Skip if there's zero variance
            if density_data.var() == 0:
                continue

            # Check for NaN or Infinite Values
            if np.any(~np.isfinite(density_data)):
                print(f'Found non-finite values in {numeric_column} for {class_name}.')
                continue

            hist, edges = np.histogram(density_data, density=True, bins=50)
            density_plot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=color, line_color='#00000000', alpha=0.7, legend_label=f'{numeric_column}')

        # Customize the density plot (optional)
        density_plot.axis.major_label_text_color = '#fff'
        density_plot.xaxis.major_label_text_font_size = '10pt'
        density_plot.yaxis.major_label_text_font_size = '10pt'
        density_plot.legend.label_text_font_size = '10pt'
        density_plot.border_fill_color = '#00000000' 
        density_plot.xgrid.grid_line_color = None
        density_plot.ygrid.grid_line_color = None
        density_plot.legend.background_fill_color = '#00000000'
        density_plot.legend.border_line_color = "#00000000"
        density_plot.legend.label_text_color = '#fff'

        density_plots.append(density_plot)
    
    # Create a layout for density plots with two columns
    density_plots_layout = column(
        row(density_plots[0], density_plots[1], density_plots[2]),
        row(density_plots[3], density_plots[4], density_plots[5]),
        row(density_plots[6])
    )

    # Create Bokeh components for density plots
    scripts_density, divs_density = components(density_plots_layout)
    
    # Calculate the correlation matrix
    correlations = data.corr(method='pearson', numeric_only=True)
    
    # Get the upper triangular part of the correlation matrix
    matrix = np.triu(correlations)

    reversed_y_categories = list(reversed(correlations.columns))

    # Create a Bokeh figure
    relations = figure(tools="hover", height=350, width=600, background_fill_color='rgba(0, 0, 0, 0)', x_range=list(correlations.index), y_range=reversed_y_categories)

    # Create a color mapper
    original_cmap = ['#FFC959', '#FF9B56', '#ED715E', '#CE4D68', '#A2346F', '#8753A0', '#4C70BD', '#0086C1', '#0096B0', '#1AA196']
    mapper = linear_cmap(field_name='values', palette=original_cmap, low=-1, high=1)

    # Convert the correlation matrix to a long format
    data = {'row': [], 'col': [], 'values': []}
    for i, rows in enumerate(correlations.index):
        for j, col in enumerate(correlations.columns):
            if matrix[i, j] != 0:
                data['row'].append(rows)
                data['col'].append(col)
                data['values'].append(correlations.iloc[i, j])

    print(data['values'])
    source = pd.DataFrame(data)
    source['values'] = source['values'].round(2)

    # Plot rectangles for the correlation matrix
    relations.rect(x='row', y='col', width=1, height=1, source=source,
                fill_color=mapper, line_color='white', line_width=2)

    # Annotate the squares with correlation values
    relations.text(x='row', y='col', text='values', source=source,
                text_color='black', text_align='center', text_baseline='middle')

    # Add color bar
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0, 0))
    color_bar.background_fill_color = 'rgba(0, 0, 0, 0)'
    relations.add_layout(color_bar, 'right')

    # Customize plot properties
    relations.axis.major_label_text_color = '#fff'
    relations.title.text_font_size = "16px"
    relations.axis.major_label_standoff = 12
    relations.xaxis.major_label_orientation = 1
    relations.border_fill_color = '#00000000' 
    relations.xgrid.grid_line_color = None
    relations.ygrid.grid_line_color = None

    # Create Bokeh components for correlation matrix plot
    script_relations, div_relations = components(relations)

    return render(request, 'eda.html', {
        "data_head_json": json.loads(data_head_json),
        'script_bar': script_bar, 
        'div_bar': div_bar,
        'scripts_density': scripts_density,
        'divs_density': divs_density,
        'script_relations': script_relations,
        'div_relations': div_relations,
    })

def train_view(request):
    return render (request, 'train.html')

def train_model(request):
     
    if request.method == 'POST':
        store = firestore.client()
        doc_ref = store.collection(u'rice')

        df = []

        docs = doc_ref.get()
        for doc in docs:
            df.append(doc.to_dict())

        # Convert the list of dictionaries into a Pandas DataFrame
        dataframe = pd.DataFrame(df)
        # Convert data types
        dataframe['Extent'] = dataframe['Extent'].astype(float)
        dataframe['Perimeter'] = dataframe['Perimeter'].astype(float)
        dataframe['Convex_Area'] = dataframe['Convex_Area'].astype(int)
        dataframe['Eccentricity'] = dataframe['Eccentricity'].astype(float)
        dataframe['Major_Axis_Length'] = dataframe['Major_Axis_Length'].astype(float)
        dataframe['Minor_Axis_Length'] = dataframe['Minor_Axis_Length'].astype(float)
        dataframe['Area'] = dataframe['Area'].astype(int)

        
        label_encoder = LabelEncoder()
        dataframe['Class'] = label_encoder.fit_transform(dataframe['Class'])
        Y = dataframe['Class'].values  # Use 'Class' as the target variable

        # Extract features excluding 'Class'
        X = dataframe.drop(['Class'], axis=1).values

        num_folds = int(request.POST.get('num_fold'))
        seed = int(request.POST.get('random_state'))
        c = float(request.POST.get('c'))

        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        model = LogisticRegression(max_iter=200, random_state=seed, solver='lbfgs', C=c)

        accuracy_scores = []

        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            model.fit(X_train, Y_train)

            # Evaluate the model on the test set and store the accuracy
            Y_pred = model.predict(X_test)
            accuracy = accuracy_score(Y_test, Y_pred)
            accuracy_scores.append(accuracy)

        # Specify the model file path
        model_directory = 'C:\\Users\\Acer\\Desktop\\itd105\\machinelearning\\models'
        os.makedirs(model_directory, exist_ok=True)  # Create the directory if it doesn't exist
        model_filename = os.path.join(model_directory, 'trained_model.joblib')

        # Delete the old model file if it exists
        if os.path.exists(model_filename):
            os.remove(model_filename)

        # Save the trained model
        joblib.dump((model, label_encoder), model_filename)

        # Calculate and print the mean accuracy
        mean_accuracy = round(((sum(accuracy_scores) / num_folds) * 100), 2)
        print(f'Mean Accuracy: {mean_accuracy}')

        return render(request, 'train.html', {'accuracy_scores': mean_accuracy})
     
def predict_view(request):
    return render (request, 'predict.html')

def postPredict(request):
    if request.method == 'POST':
        # Get user input from the form
        user_input = {
            'Extent': float(request.POST.get('extent')),
            'Perimeter': float(request.POST.get('perimeter')),
            'Convex_Area': int(request.POST.get('convex_area')),
            'Eccentricity': float(request.POST.get('eccentricity')),
            'Major_Axis_Length': float(request.POST.get('major_axis_length')),
            'Minor_Axis_Length': float(request.POST.get('minor_axis_length')),
            'Area': int(request.POST.get('area')),
        }

        # Create a DataFrame from the user input
        input_df = pd.DataFrame([user_input])

        # Define the path to the trained model
        model_directory = 'C:\\Users\\Acer\\Desktop\\itd105\\machinelearning\\models'
        model_filename = os.path.join(model_directory, 'trained_model.joblib')

        # Load the trained model and label encoder
        model, label_encoder = joblib.load(model_filename)


        # Make predictions using the loaded model
        input_features = input_df.values
        predicted_class = model.predict(input_features)[0]

        # Convert the predicted class back to the original label
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]

        # Pass the predicted label to the template
        print(predicted_label)
        return JsonResponse({'predicted_label': predicted_label})

    # Render the form for user input
    return render(request, 'predict.html')

def add_data(request):
    return render (request, 'add_data.html')

def addData (request):
    if request.method == 'POST':

        # Save data to Firestore
        data = {
            "Class": request.POST.get('class'),
            "Extent": request.POST.get('extent'),
            "Perimeter":request.POST.get('perimeter'),
            "Convex_Area": request.POST.get('convex_area'),
            "Eccentricity": request.POST.get('eccentricity'),
            "Major_Axis_Length": request.POST.get('major_axis_length'),
            "Minor_Axis_Length": request.POST.get('minor_axis_length'),
            "Area": request.POST.get('area')
        }
        firestore.client().collection('rice').add(data)
        message = "Success"
    return render (request, "add_data.html", {'message':message})

def export (request):

    store = firestore.client()
    doc_ref = store.collection(u'rice')

    df = []

    docs = doc_ref.get()
    for doc in docs:
        df.append(doc.to_dict())

    data = pd.DataFrame(df)
    
    # Save DataFrame as CSV
    data.to_csv('rice_data.csv', index=False)

    # Open the file in binary mode for reading
    with open('rice_data.csv', 'rb') as csv_file:
        response = HttpResponse(csv_file.read(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=rice_data.csv'

    # Return the response
    return response


def home_reg (request):
    return render (request, "home_reg.html")

def eda_reg(request):
    store = firestore.client()
    doc_ref = store.collection(u'productivity')

    df = []

    docs = doc_ref.get()
    for doc in docs:
        df.append(doc.to_dict())

    # Convert the list of dictionaries into a Pandas DataFrame
    data = pd.DataFrame(df)
    data_head_json = data.to_json(orient='split')

    # Convert numeric columns to numeric type
    numeric_columns = ['team', 'days', 'quart', 'dept', 'targeted_productivity', 'smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers', 'actual_productivity']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Loop through each numeric column and create a density plot
    density_plots = []
    #creating color map
    original_cmap = ['#A2346F', '#8563AE', '#004F7C', '#e31a1c', '#A2346F', '#8D7257', '#374955', '#007260', '#004F7C', '#A2346F', '#e31a1c', '#8563AE', '#8D7257', '#374955']

    for numeric_column, color in zip(numeric_columns, original_cmap):
        density_plot = figure(title=f'{numeric_column}', height=300, width=335, background_fill_color="rgba(0, 0, 0, 0)", margin=(10, 10, 10, 10), toolbar_location=None, tools="hover")
        density_plot.title.text_color = "#fff"

        # Create kernel density estimate
        values = data[numeric_column].dropna()
        kde = gaussian_kde(values)
        x = np.linspace(values.min(), values.max(), 1000)
        y = kde(x)
        
        # Plot the density curve
        density_plot.line(x, y, line_color=color, line_width=2, legend_label=f'{numeric_column}')

        # Add shading to the area under the density curve
        band = Band(base='x', lower='y', source=ColumnDataSource(data={'x': x, 'y': y}), level='underlay',
                    fill_alpha=0.5, line_width=1, line_color=color)
        density_plot.add_layout(band)

        # Customize the density plot (optional)
        density_plot.axis.major_label_text_color = '#fff'
        density_plot.xaxis.major_label_text_font_size = '10pt'
        density_plot.yaxis.major_label_text_font_size = '12pt'
        density_plot.legend.label_text_font_size = '12pt'
        density_plot.border_fill_color = '#00000000' 
        density_plot.xgrid.grid_line_color = None
        density_plot.ygrid.grid_line_color = None
        density_plot.legend.background_fill_color = '#00000000'
        density_plot.legend.border_line_color = "#00000000"
        density_plot.legend.label_text_color = '#fff'

        density_plots.append(density_plot)
    
    # Create a layout for density plots with two columns
    density_plots_layout = column(
        row(density_plots[0], density_plots[1], density_plots[2]),
        row(density_plots[3], density_plots[4], density_plots[5]),
        row(density_plots[6], density_plots[7], density_plots[8]),
        row(density_plots[9], density_plots[10], density_plots[11]),
        row(density_plots[12], density_plots[1])
    )

    # Create Bokeh components for density plots
    scripts_density, divs_density = components(density_plots_layout)
    
    # Calculate the correlation matrix
    correlations = data.corr(method='pearson', numeric_only=True)
    

    # Get the upper triangular part of the correlation matrix
    matrix = np.triu(correlations)

    reversed_y_categories = list(reversed(correlations.columns))

    # Create a Bokeh figure
    relations = figure(tools="hover", height=800, width=1070, background_fill_color='rgba(0, 0, 0, 0)', x_range=list(correlations.index), y_range=reversed_y_categories)

    # Create a color mapper
    original_cmap = ['#FFC959', '#FF9B56', '#ED715E', '#CE4D68', '#A2346F', '#8753A0', '#4C70BD', '#0086C1', '#0096B0', '#1AA196']
    mapper = linear_cmap(field_name='values', palette=original_cmap, low=-1, high=1)

    # Convert the correlation matrix to a long format
    data = {'row': [], 'col': [], 'values': []}
    print('cor index', correlations.index)
    print('cor column', correlations.columns)
    for i, rows in enumerate(correlations.index):
        for j, col in enumerate(correlations.columns):
            if matrix[i, j] != 0:
                data['row'].append(rows)
                data['col'].append(col)
                data['values'].append(correlations.iloc[i, j])
    source = pd.DataFrame(data)
    source['values'] = source['values'].round(2)

    # Plot rectangles for the correlation matrix
    relations.rect(x='row', y='col', width=1, height=1, source=source,
                fill_color=mapper, line_color='white', line_width=1)

    # Annotate the squares with correlation values
    relations.text(x='row', y='col', text='values', source=source,
                text_color='black', text_align='center', text_baseline='middle')


    # Add color bar
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0, 0))
    color_bar.background_fill_color = 'rgba(0, 0, 0, 0)'
    relations.add_layout(color_bar, 'right')

    # Customize plot properties
    relations.axis.major_label_text_color = '#fff'
    relations.title.text_font_size = "16px"
    relations.axis.major_label_standoff = 12
    relations.xaxis.major_label_orientation = 1
    relations.xaxis.major_label_text_font_size = '12pt'
    relations.yaxis.major_label_text_font_size = '12pt'
    relations.border_fill_color = '#00000000' 
    relations.xgrid.grid_line_color = None
    relations.ygrid.grid_line_color = None

    # Create Bokeh components for correlation matrix plot
    script_relations, div_relations = components(relations)

    return render(request, 'eda_reg.html', {
        "data_head_json": json.loads(data_head_json),
        'scripts_density': scripts_density,
        'divs_density': divs_density,
        'script_relations': script_relations,
        'div_relations': div_relations,
    })

def add_data_reg(request):
    return render (request, "add_data_reg.html")

def addData_reg(request):
     if request.method == 'POST':

        # Save data to Firestore
        data = {
            "team": request.POST.get('team'),
            "targeted_productivity": request.POST.get('targeted_productivity'),
            "smv":request.POST.get('smv'),
            "wip": request.POST.get('wip'),
            "over_time": request.POST.get('over_time'),
            "incentive": request.POST.get('incentive'),
            "idle_time": request.POST.get('idle_time'),
            "idle_men": request.POST.get('idle_men'),
            "no_of_style_change": request.POST.get('no_of_style_change'),
            "no_of_workers": request.POST.get('no_of_workers'),
            "actual_productivity": request.POST.get('actual_productivity'),
            "days": request.POST.get('days'),
            "quart": request.POST.get('quart'),
            "dept": request.POST.get('dept')
        }
        firestore.client().collection('productivity').add(data)
        message = "Success"
        return render (request, "add_data_reg.html", {'message':message})

def export_reg(request):
    store = firestore.client()
    doc_ref = store.collection(u'productivity')

    df = []

    docs = doc_ref.get()
    for doc in docs:
        df.append(doc.to_dict())

    data = pd.DataFrame(df)
    
    # Save DataFrame as CSV
    data.to_csv('productivity_data.csv', index=False)

    # Open the file in binary mode for reading
    with open('productivity_data.csv', 'rb') as csv_file:
        response = HttpResponse(csv_file.read(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=productivity_data.csv'

    # Return the response
    return response

def train_model_reg(request):
    return render (request, "train_reg.html")

def postTrain_reg(request):
        store = firestore.client()
        doc_ref = store.collection(u'productivity')

        df = []

        docs = doc_ref.get()
        for doc in docs:
            df.append(doc.to_dict())

        dataframe = pd.DataFrame(df)

        Y = dataframe['actual_productivity'].values 

        # Extract features, excluding 'actual_productivity'
        X = dataframe.drop(['actual_productivity'], axis=1)

        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        print(X.columns)

        test_size = 0.20
        seed = 10

        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=test_size, random_state=seed)
        # Train the data on a Linear Regression model
        linear_model = LinearRegression()

        # Train the model on the training data
        linear_model.fit(X_train, Y_train)

        # Make predictions on the test data
        Y_pred = linear_model.predict(X_test)

        # Specify the model file path
        model_directory = 'C:\\Users\\Acer\\Desktop\\itd105\\machinelearning\\models'
        os.makedirs(model_directory, exist_ok=True)  # Create the directory if it doesn't exist
        model_filename = os.path.join(model_directory, 'trained_reg_model.joblib')

        # Delete the old model file if it exists
        if os.path.exists(model_filename):
            os.remove(model_filename)

        #save model
        joblib.dump(linear_model, model_filename)

        # Save the scaler for future use during prediction
        scaler_filename = os.path.join(model_directory, 'scaler.joblib')
        joblib.dump(scaler, scaler_filename)

        # Calculate the mean absolute error (MAE) for the predictions
        mae = mean_absolute_error(Y_test, Y_pred)
        mae_final = "%.3f" % mae
        print("MAE: %.3f" % mae)
        return render(request, 'train_reg.html', {'accuracy_scores': mae_final})

def predict_model_reg(request):
    return render(request, "predict_reg.html")

def postPredict_reg(request):
    if request.method == 'POST':
        # Extract user input from the POST request
        user_input = {
            'smv': float(request.POST.get('smv')),
            'targeted_productivity': float(request.POST.get('targeted_productivity')),
            'quart': int(request.POST.get('quart')),
            'no_of_style_change': float(request.POST.get('no_of_style_change')),
            'dept': int(request.POST.get('dept')),
            'idle_men': float(request.POST.get('idle_men')),
            'team': float(request.POST.get('team')),
            'wip': float(request.POST.get('wip')),
            'over_time': float(request.POST.get('over_time')),
            'incentive': int(request.POST.get('incentive')),
            'idle_time': float(request.POST.get('idle_time')),
            'days': int(request.POST.get('days')),
            'no_of_workers': float(request.POST.get('no_of_workers')),
        }

        # Create a DataFrame from the user input
        input_df = pd.DataFrame([user_input])

        # Define the path to the trained model
        model_directory = 'C:\\Users\\Acer\\Desktop\\itd105\\machinelearning\\models'
        model_filename = os.path.join(model_directory, 'trained_reg_model.joblib')

        # Check if the model file exists
        if not os.path.exists(model_filename):
            return JsonResponse({'error': 'Model file not found'})

        # Load the trained model
        model = joblib.load(model_filename)
        print(model)

        # Load the StandardScaler used during training
        scaler_filename = os.path.join(model_directory, 'scaler.joblib')
        scaler = joblib.load(scaler_filename)

        # Apply feature scaling to user input
        scaled_input = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

        # Make predictions using the trained model
        prediction = model.predict(scaled_input)
        print(prediction)

        # Convert NumPy array to Python list
        prediction_list = prediction.tolist()
        print(prediction_list)

        # Pass the predicted label to the template as a JSON response
        return JsonResponse({'predicted_label': prediction_list})

    # Render the form for user input
    return render(request, 'predict_reg.html')
