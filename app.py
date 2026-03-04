"""
Streamlit application for Document Intelligence Chatbot.
"""
import streamlit as st
import os
from pathlib import Path

# Import custom modules
from config.settings import validate_config, TEMP_UPLOAD_DIR, USE_RETRIEVAL
from core.azure_doc_service import AzureDocumentService
from utils.image_utils import pdf_to_images, create_region_images
from utils.visualization import visualize_layout, display_region_grid, create_summary_stats
from agents.document_agent import DocumentAgent
from core.retriever import DocumentRetriever

# Page configuration
st.set_page_config(
    page_title="Document Intelligence Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""

""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    if 'pdf_path' not in st.session_state:
        st.session_state.pdf_path = None
    if 'page_images' not in st.session_state:
        st.session_state.page_images = []
    if 'ocr_regions' not in st.session_state:
        st.session_state.ocr_regions = []
    if 'layout_regions' not in st.session_state:
        st.session_state.layout_regions = []
    if 'region_images' not in st.session_state:
        st.session_state.region_images = {}
    if 'ordered_text' not in st.session_state:
        st.session_state.ordered_text = []
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'stats' not in st.session_state:
        st.session_state.stats = {}


init_session_state()

# Helper Functions
def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temp directory."""
    file_path = os.path.join(TEMP_UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def process_document(pdf_path: str):
    """Process the uploaded document."""
    try:
        with st.spinner("🔄 Processing document... This may take a few minutes."):

            # Step 1: Convert PDF to images
            st.info("📄 Converting PDF to images...")
            page_images = pdf_to_images(pdf_path)
            st.session_state.page_images = page_images
            total_pages = len(page_images)

            # Step 2: Analyze with Azure Document Intelligence
            st.info(f"🔍 Analyzing {total_pages} pages with Azure Document Intelligence...")
            doc_service = AzureDocumentService()
            ocr_regions, layout_regions = doc_service.analyze_document(pdf_path, total_pages)
            st.session_state.ocr_regions = ocr_regions
            st.session_state.layout_regions = layout_regions

            # Step 3: Get ordered text
            st.info("📝 Extracting ordered text...")
            ordered_text = doc_service.get_ordered_text()
            st.session_state.ordered_text = ordered_text

            # Step 4: Create region images
            st.info("🖼️ Creating region images...")
            region_images = create_region_images(layout_regions, page_images)
            st.session_state.region_images = region_images

            # ── Step 5 (NEW): Build retriever ──
            retriever = None
            if USE_RETRIEVAL:
                st.info("🔎 Building semantic search index (Chroma)...")
                try:
                    retriever = DocumentRetriever(
                        ordered_text=ordered_text,
                        layout_regions=layout_regions,
                    )
                    st.session_state.retriever = retriever
                    st.success(
                        f"✅ Retrieval index built — "
                        f"{retriever.get_stats()['total_chunks']} chunks indexed"
                    )
                except Exception as e:
                    st.warning(
                        f"⚠️ Retrieval setup failed ({e}). "
                        f"Falling back to full-context mode."
                    )
                    retriever = None

            # Step 6: Initialize agent
            st.info("🤖 Initializing AI agent...")
            agent = DocumentAgent(
                ocr_regions=ocr_regions,
                layout_regions=layout_regions,
                region_images=region_images,
                ordered_text=ordered_text,
                retriever=retriever,                # ← NEW
            )
            st.session_state.agent = agent

            # Step 7: Get statistics
            st.session_state.stats = create_summary_stats(ocr_regions, layout_regions)

            st.session_state.document_processed = True
            st.session_state.pdf_path = pdf_path

            st.success("✅ Document processed successfully!")
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        st.exception(e)


def reset_session():
    """Reset session state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()
    st.rerun()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/documents.png", width=80)
    st.title("📄 Document Intelligence")
    st.markdown("---")

    # Validate configuration
    try:
        validate_config()
        st.success("✅ Configuration valid")
    except ValueError as e:
        st.error(f"❌ Configuration error: {str(e)}")
        st.stop()

    st.markdown("### Upload Document")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document with text, tables, and charts"
    )

    if uploaded_file is not None:
        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            # Save and process
            pdf_path = save_uploaded_file(uploaded_file)
            process_document(pdf_path)

    # Show processing status
    if st.session_state.document_processed:
        st.markdown("---")
        st.markdown("### 📊 Document Statistics")

        stats = st.session_state.stats

        st.markdown(f"""
        

        Pages: {stats.get('pages_with_content', 0)}

        Text Regions: {stats.get('total_ocr_regions', 0)}

        Tables: {stats.get('tables', 0)}

        Figures: {stats.get('figures', 0)}

        Text Blocks: {stats.get('text_blocks', 0)}

        Avg Confidence: {stats.get('avg_confidence', 0):.2%}
        

        """, unsafe_allow_html=True)

        st.markdown("---")

        # Reset button
        if st.button("🔄 Process New Document", use_container_width=True):
            reset_session()

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This chatbot uses:
    - **Azure Document Intelligence** for OCR and layout detection
    - **Azure OpenAI** for visual analysis and conversation
    - **LangChain** for agent orchestration

    Ask questions about text, tables, and charts in your document!
    """)

# Main Content Area
st.markdown("""
🤖 Document Intelligence Chatbot
""", unsafe_allow_html=True)

if not st.session_state.document_processed:
    # Welcome screen
    st.markdown("""
    ## Welcome! 👋

    Upload a PDF document to get started. This AI-powered chatbot can:

    - 📝 **Extract and understand text** in reading order
    - 📊 **Analyze tables** and extract structured data
    - 📈 **Interpret charts and graphs** with specific data points
    - 💬 **Answer questions** by combining information from text, tables, and visuals

    ### How to use:
    1. Upload a PDF document using the sidebar
    2. Click "Process Document" and wait for analysis to complete
    3. Ask questions in the chat interface

    ### Example questions:
    - "What is this document about?"
    - "Extract all data from the tables"
    - "What trends are shown in the charts?"
    - "Summarize the key findings with specific numbers"
    """)

    st.info("👈 Upload a document from the sidebar to begin!")

else:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📄 Document View", "📊 Extracted Data", "🔍 Debug"])

    # Tab 1: Chat Interface
    with tab1:
        st.markdown("### Chat with your document")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get agent response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.agent.query(prompt)
                    st.markdown(response)

            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Suggested questions
        if len(st.session_state.messages) == 0:
            st.markdown("#### 💡 Suggested questions:")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📝 What is this document about?"):
                    st.session_state.messages.append({"role": "user", "content": "What is this document about?"})
                    st.rerun()

                if st.button("📊 Extract data from tables"):
                    st.session_state.messages.append({"role": "user", "content": "Extract all data from the tables in this document."})
                    st.rerun()

            with col2:
                if st.button("📈 Analyze charts and graphs"):
                    st.session_state.messages.append({"role": "user", "content": "Analyze all charts and graphs. What trends do they show?"})
                    st.rerun()

                if st.button("📋 Summarize key findings"):
                    st.session_state.messages.append({"role": "user", "content": "Summarize the key findings from this document with specific data."})
                    st.rerun()
    # Tab 2: Document View
    with tab2:
        st.markdown("### Document Layout Detection")

        # Page selector
        page_num = st.selectbox(
            "Select page to view:",
            range(1, len(st.session_state.page_images) + 1),
            format_func=lambda x: f"Page {x}"
        )

        if st.session_state.page_images:
            # Visualize layout
            fig = visualize_layout(
                st.session_state.page_images[page_num - 1],
                st.session_state.layout_regions,
                page_num=page_num,
                title=f"Layout Detection - Page {page_num}"
            )
            st.pyplot(fig)

            # Show regions on this page
            page_regions = [r for r in st.session_state.layout_regions if r.page_number == page_num]

            st.markdown(f"#### Regions on Page {page_num}")
            for region in page_regions:
                st.markdown(f"- **Region {region.region_id}**: {region.region_type}")

    # Tab 3: Extracted Data
    with tab3:
        st.markdown("### Extracted Tables and Figures")

        # Filter by type
        data_type = st.radio(
            "Select type:",
            ["All", "Tables", "Figures"],
            horizontal=True
        )

        # Filter regions
        if data_type == "Tables":
            filtered_regions = [r for r in st.session_state.layout_regions if r.region_type == 'table']
        elif data_type == "Figures":
            filtered_regions = [r for r in st.session_state.layout_regions if r.region_type == 'figure']
        else:
            filtered_regions = [r for r in st.session_state.layout_regions if r.region_type in ['table', 'figure']]

        if filtered_regions:
            # Display cropped regions
            st.markdown(f"#### Found {len(filtered_regions)} {data_type.lower()}")

            for region in filtered_regions:
                with st.expander(f"Region {region.region_id}: {region.region_type.capitalize()} (Page {region.page_number})"):
                    if region.region_id in st.session_state.region_images:
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.image(
                                st.session_state.region_images[region.region_id]['image'],
                                caption=f"Region {region.region_id}",
                                use_container_width=True
                            )

                        with col2:
                            st.markdown("**Details:**")
                            st.markdown(f"- **Type:** {region.region_type}")
                            st.markdown(f"- **Page:** {region.page_number}")
                            st.markdown(f"- **Confidence:** {region.confidence:.2%}")

                            if region.content:
                                if region.region_type == 'table':
                                    st.markdown(f"- **Dimensions:** {region.content['row_count']}x{region.content['column_count']}")
                                elif region.region_type == 'figure' and region.content.get('caption'):
                                    st.markdown(f"- **Caption:** {region.content['caption']}")
        else:
            st.info(f"No {data_type.lower()} found in this document.")

    # Tab 4: Debug
    with tab4:
        st.markdown("### Debug Information")

        # Statistics
        st.markdown("#### Document Statistics")
        st.json(st.session_state.stats)

        # OCR Regions
        with st.expander("📝 OCR Regions (First 20)"):
            for i, region in enumerate(st.session_state.ocr_regions[:20]):
                st.markdown(f"**[{i}]** Page {region.page_number}: {region.text[:100]}...")

        # Layout Regions
        with st.expander("🏗️ Layout Regions"):
            for region in st.session_state.layout_regions[:20]:
                st.markdown(f"- **Region {region.region_id}**: {region.region_type} on page {region.page_number}")

        # Agent Info
        with st.expander("🤖 Agent Information"):
            if st.session_state.agent:
                agent_stats = st.session_state.agent.get_statistics()
                st.json(agent_stats)

        # Raw ordered text
        with st.expander("📄 Ordered Text (First 50)"):
            for item in st.session_state.ordered_text[:50]:
                st.markdown(f"[{item['position']}] {item['text']}")
# Footer
st.markdown("---")
st.markdown("""
    

Powered by Azure Document Intelligence & Azure OpenAI


    

Built with Streamlit & LangChain




""", unsafe_allow_html=True)


# Main execution
if __name__ == "__main__":
    pass  # Streamlit handles the execution
