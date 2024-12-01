"""Models for FAERS data structures."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

@dataclass
class Demographics:
    """Represents demographic information from FAERS."""
    primary_id: str
    case_id: str
    quarter: str
    case_version: Optional[str] = None
    i_f_code: Optional[str] = None
    sex: Optional[str] = None
    age: Optional[float] = None
    age_code: Optional[str] = None
    weight: Optional[float] = None
    weight_code: Optional[str] = None
    reporter_country: Optional[str] = None
    occurrence_country: Optional[str] = None
    event_date: Optional[datetime] = None
    report_date: Optional[datetime] = None

@dataclass
class Drug:
    """Represents drug information from FAERS."""
    primary_id: str
    drug_seq: str
    role_code: str
    drug_name: str
    substance: Optional[str] = None
    prod_ai: Optional[str] = None

@dataclass
class DrugInfo:
    """Additional drug information model."""
    primary_id: str
    drug_seq: str
    val_vbm: Optional[str] = None
    nda_num: Optional[str] = None
    lot_num: Optional[str] = None
    route: Optional[str] = None
    dose_form: Optional[str] = None
    dose_freq: Optional[str] = None
    exp_dt: Optional[datetime] = None
    dose_amt: Optional[float] = None
    dose_unit: Optional[str] = None
    dechal: Optional[str] = None
    rechal: Optional[str] = None

@dataclass
class Indication:
    """Drug indication model."""
    primary_id: str
    drug_seq: str
    indi_pt: str

@dataclass
class Reaction:
    """Represents adverse reaction information from FAERS."""
    primary_id: str
    pt: str  # Preferred Term
    drug_rec_act: Optional[str] = None

@dataclass
class Outcome:
    """Represents outcome information from FAERS."""
    primary_id: str
    outcome_code: str

@dataclass
class ReportSource:
    """Report source model."""
    primary_id: str
    rpsr_cod: str

@dataclass
class Therapy:
    """Drug therapy model."""
    primary_id: str
    drug_seq: str
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    dur: Optional[float] = None
    dur_cod: Optional[str] = None
