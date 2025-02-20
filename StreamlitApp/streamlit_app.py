import streamlit as st
from commute_analysis import main as process_commute_data

# Add file uploader in Streamlit
uploaded_files = st.file_uploader("Upload commute data files", accept_multiple_files=True)

if uploaded_files:
    # Create temporary directory for uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, 'wb') as f:
                f.write(file.getbuffer())
        
        # Process the data
        processed_data = process_commute_data(temp_dir)
        
        # Display the results
        st.dataframe(processed_data)