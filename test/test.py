import streamlit as st

st.title("StreamLit Intro By Santhosh")
st.write("This is a sample work on learning how to use Streamlit")
if 'tasks' not in st.session_state:
    st.session_state.tasks=[]
task=st.text_input("Enter the task")
st.selectbox("Task type",("Yes or No","measurable"))
if st.button("Add task") and task:
    st.session_state.tasks.append(task)
st.title("Task Panel")

for i in len(st.session_state.tasks):
    st.write(st.session_state.tasks[i])
    if st.button("Delete"):
        st.session_state.tasks.remove(i)