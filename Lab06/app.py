import streamlit as st
import pickle as pkl
from datetime import datetime

startTime = datetime.now()

filename = "model.h5"
model = pickle.load(open(filename, 'rb'))