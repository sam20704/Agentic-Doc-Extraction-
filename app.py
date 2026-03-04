from config.settings import validate_config, TEMP_UPLOAD_DIR, USE_RETRIEVAL
from core.retriever import DocumentRetriever          # ← NEW import


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
