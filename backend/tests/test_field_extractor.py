import pytest
from app.services.field_extractor import FieldExtractor

def test_extract_cvc_genuine_text():
    sample_text = """
    GOVERNMENT OF MAHARASHTRA
    Social Justice and Special Assistance Department
    CERTIFICATE OF VALIDITY
    DISTRICT CASTE CERTIFICATE SCRUTINY COMMITTEE, SANGLI DISTRICT
    Committee Decision No. - SNG/EDU/2025/10298
    Date - 18/07/2025
    VC No.: 2025-001656685
    WHEREAS, an application of Bapu Suresh Nikam dated - 16/07/2025 was placed before Committee.
    The Committee has decided that the Caste Certificate is VALID.
    Caste: Maratha (Socially and Educationally Backward Class)
    """
    fields = FieldExtractor.extract_fields(
        predicted_class="CASTE_VALIDITY_CERTIFICATE",
        full_text=sample_text,
        filename="UG25000837_BAPU SURESH NIKAM_CVC.pdf"
    )

    assert fields["predicted_class"] == "CASTE_VALIDITY_CERTIFICATE"
    assert fields["enrollment_number"] == "UG25000837"
    assert "BAPU SURESH NIKAM" in fields["candidate_name"]
    assert "2025-001656685" in fields["certificate_no"]
    assert "SNG/EDU/2025/10298" in fields["decision_no"]
    assert fields["validity_decision"] == "VALID"
    assert fields["caste_claim"] == "MARATHA"
    assert fields["district"] == "Sangli"

def test_extract_cvr_text():
    sample_text = """
    District Caste Certificate Scrutiny Committee Kolhapur
    Application Number: ED-2025-01993874
    Receipt No: 25-26-KOL-2197570
    Applicant's Name: Miss. Tasnim Jabbar Patel
    Received on: 30/07/2025
    Caste: Muslim Bagwan
    """
    fields = FieldExtractor.extract_fields(
        predicted_class="CASTE_VALIDITY_RECEIPT",
        full_text=sample_text,
        filename="252000108_TASNIM JABBAR PATEL_CVR.pdf"
    )

    assert fields["predicted_class"] == "CASTE_VALIDITY_RECEIPT"
    assert fields["application_no"] == "ED-2025-01993874"
    assert fields["receipt_no"] == "25-26-KOL-2197570"
    assert fields["received_date"] == "30/07/2025"
    assert fields["district"] == "Kolhapur"

def test_extract_leaving_certificate():
    sample_text = """
    SHUBHAMRAJE JR. COLLEGE
    LEAVING CERTIFICATE
    Serial No.: 0859
    G.R. No.: 527
    Student Name: JADHAV SAMRUDDH SAGAR
    Mother Tongue: MARATHI
    Religion: HINDU
    Date of birth: 23/09/2006
    """
    fields = FieldExtractor.extract_fields(
        predicted_class="LEAVING_CERTIFICATE",
        full_text=sample_text,
        filename="EN20260020_JADHAV SAMRUDDH SAGAR_LC.pdf"
    )

    assert fields["serial_no"] == "0859"
    assert fields["gr_no"] == "527"
    assert fields["dob"] == "23/09/2006"
    assert fields["mother_tongue"] == "Marathi"
    assert fields["religion"] == "Hindu"

def test_extract_proforma_o():
    sample_text = """
    PROFORMA - O
    Minority Community Student's Self Declaration
    I, Samruddh Sagar Jadhav, declare that I belong to the
    Linguistic Minority Community (Sindhi).
    Dated: 2024
    """
    fields = FieldExtractor.extract_fields(
        predicted_class="PROFORMA_O",
        full_text=sample_text,
        filename="EN20260020_Samruddh Sagar Jadhav_Proforma-O.pdf"
    )

    assert fields["minority_type"] == "Linguistic Minority"
    assert fields["community_mother_tongue"] == "Sindhi"

def test_extract_caste_certificate():
    sample_text = """
    GOVERNMENT OF MAHARASHTRA
    OFFICE OF THE SUB-DIVISIONAL OFFICER
    CASTE CERTIFICATE
    Certificate No: MRC/CASTE/2024/78219
    Date: 15/06/2023
    This is to certify that Shri Rahul Sharma belongs to Kunbi Maratha (OBC).
    """
    fields = FieldExtractor.extract_fields(
        predicted_class="CASTE_CERTIFICATE",
        full_text=sample_text,
        filename="EN25272514_Rahul Sharma_Cast certificate.pdf"
    )

    assert fields["certificate_no"] == "MRC/CASTE/2024/78219"
    assert fields["issued_date"] == "15/06/2023"
    assert fields["issuing_authority"] == "Sub-Divisional Officer"
    assert "KUNBI" in fields["caste_claim"]

def test_extract_bilingual_marathi_fields():
    # Test CVC with Marathi and bilingual headers/fields
    sample_text = """
    महाराष्ट्र शासन
    सामाजिक न्याय व विशेष सहाय्य विभाग
    जात प्रमाणपत्र पडताळणी समिती, सातारा
    प्रमाणपत्र क्रमांक: 2025-001648641
    निर्णय क्रमांक: 11992
    दिनांक: 18/07/2025
    जात प्रमाणपत्र वैध करण्यात येत आहे.
    जात प्रवर्ग: मराठा कुणबी
    """
    fields = FieldExtractor.extract_fields(
        predicted_class="CASTE_VALIDITY_CERTIFICATE",
        full_text=sample_text,
        filename="UG25000837_RUDRAKSHA SANJAY GHADGE_CVC.pdf"
    )

    assert fields["certificate_no"] == "2025-001648641"
    assert fields["decision_no"] == "11992"
    assert fields["issued_date"] == "18/07/2025"
    assert fields["validity_decision"] == "VALID"
    assert fields["caste_claim"] == "MARATHA KUNBI"
    assert fields["district"] == "Satara"
    assert "Satara" in fields["committee"]
