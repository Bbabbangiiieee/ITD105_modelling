"""
URL configuration for ml project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from ml_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.signIn),
    path('postSignIn/', views.postSignIn),
    path('newuser/',views.newuser, name='newuser'),
    path('postSignUp/', views.postSignUp),
    path('/home_class/', views.home_class, name='home_class'),
    path('/eda/', views.firestore_data_view, name='eda'),
    path('/predict/', views.predict_view, name='predict'),
    path('postPredict/', views.postPredict, name='postPredict'),
    path('/train/', views.train_view, name='train'),
    path('postTrain/', views.train_model, name='postTrain'),
    path('/add_data/', views.add_data, name='add_data'),
    path('addData/', views.addData, name='addData'),
    path('/export/', views.export, name='export'),
    path('logout/', views.log_out, name='logout'),
    path('/cancel/', views.cancel, name='cancel'),

    path('/home_reg/', views.home_reg, name='home_reg'),
    path('/eda_reg/', views.eda_reg, name='eda_reg'),
    path('add_data_reg/', views.add_data_reg, name='add_data_reg'),
    path('addData_reg/', views.addData_reg, name='addData_reg'),
    path('export_reg/', views.export_reg, name='export_reg'),
    path('train_reg/', views.train_model_reg, name='train_reg'),
    path('/postTrain_reg/', views.postTrain_reg, name='postTrain_reg'),
    path('/predict_reg/', views.predict_model_reg, name='predict_reg'),
    path('postPredict_reg/', views.postPredict_reg, name='postPredict_reg'),
]
