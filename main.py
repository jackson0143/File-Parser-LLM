from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from fillpdf import fillpdfs
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
)


def create_text_embedding():
    embeddings = OpenAIEmbeddings()
    return embeddings


embedding_function = create_text_embedding()


def format_document(document):
    return "\n\n".join([doc.page_content for doc in document])


class ApplicationForm(BaseModel):
    """Model representing the expected form fields"""
    first_name: str = Field(
        default="N/A", description="First name of the applicant")
    last_name: str = Field(
        default="N/A", description="Last name of the applicant")
    email_address: str = Field(
        default="N/A", description="Email address of the applicant")
    phone_number: str = Field(
        default="N/A", description="Phone number of the applicant")
    linkedin: str = Field(default="N/A", description="LinkedIn profile URL")
    project_portfolio_url: str = Field(
        default="N/A", description="Portfolio or personal website URL")
    degree: str = Field(
        default="N/A", description="Highest degree obtained (e.g., Bachelor of Science, Master of Computer Science)")
    graduation_date: str = Field(
        default="N/A", description="Graduation date in MM/YYYY format. Look for phrases like 'graduated', 'completed degree', 'degree conferred', or dates near degree information")
    current_job_title: str = Field(
        default="N/A", description="Current job title or most recent position")
    current_employer: str = Field(
        default="N/A", description="Current employer or most recent company")
    technical_skills: str = Field(
        default="N/A", description="Technical skills, programming languages, tools, and technologies. Look for sections labeled 'Skills', 'Technical Skills', or lists of technologies")
    fit_description: str = Field(
        default="N/A", description="Description of why candidate is a good fit. Look for sections like 'Summary', 'Objective', 'About Me', or career goals")
    react_experience: str = Field(
        default="N/A", description="Years of React experience. Look for React in skills, technologies, or experience sections. If not explicitly stated, estimate based on project durations or experience descriptions")


def create_form_rag_chain(retriever, llm):
    prompt_template = ChatPromptTemplate.from_template("""
    Extract any relevant information from the provided resume to fill out an application form.
    For each field in the form, provide a clear answer if the information is available in the resume.
    If information is not available, use "N/A".
    
    Important guidelines:
    1. For dates, convert all dates to MM/YYYY format
    2. Be thorough in searching for information in all sections of the resume
 
    Resume text: {context}
    
    Extract and structure the information according to the ApplicationForm model fields.
    Pay special attention to the field descriptions in the model for specific extraction guidance.
    """)

    # Create the RAG chain with structured output
    rag_chain = (
        {"context": retriever | format_document, "question": RunnablePassthrough()}
        | prompt_template
        | llm.with_structured_output(ApplicationForm, method="function_calling")
    )

    return rag_chain


def process_pdf(input_file_path, form_file_path, output_path):
    try:
        loader = PyPDFLoader(input_file_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
        )
        retriever = vectorstore.as_retriever(search_type="similarity")

        # Get form fields
        fields = fillpdfs.get_form_fields(form_file_path)
        if not fields:
            return None, "No fillable fields found in PDF form.", None

        # Create and invoke
        form_chain = create_form_rag_chain(retriever, llm)
        form_data = form_chain.invoke(
            "Extract all form fields from the resume")
        values_dict = {}

        '''
        not sure if this can be done in a more efficient way, 
        but it is straightforward and works for the time being. 
        main focus is in the method above, so just ignore how
        i implemented this method for now
        '''
        values_dict = {
            "First name": form_data.first_name,
            "Last name": form_data.last_name,
            "email address": form_data.email_address,
            "Phone number": form_data.phone_number,
            "LinkedIn": form_data.linkedin,
            "Project Portfolio URL": form_data.project_portfolio_url,
            "Degree": form_data.degree,
            "Graduation date": form_data.graduation_date,
            "Current Job Title": form_data.current_job_title,
            "Current Employer": form_data.current_employer,
            "Technical Skills": form_data.technical_skills,
            "Describe why you are a good fit for this position": form_data.fit_description,
            "Do you have 5 years of experience in React": form_data.react_experience
        }

        # Fill and save the form to path
        fillpdfs.write_fillable_pdf(form_file_path, output_path, values_dict)

        return output_path, "Form filled successfully!", values_dict

    except Exception as e:
        return None, f"Error: {str(e)}", None


if __name__ == "__main__":
    input_file_path = "fake_resume.pdf"
    form_file_path = "Fillable fake_application_form.pdf"
    output_path = "filled_form.pdf"
    print(process_pdf(input_file_path, form_file_path, output_path))
