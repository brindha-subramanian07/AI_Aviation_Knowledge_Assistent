"""
Script to generate sample aviation documents (PDF, DOCX, TXT)
for testing and demonstration of the AI-Powered Aviation Knowledge Assistant.
"""

from pathlib import Path
import pymupdf as fitz
import docx

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def create_pdf_maintenance_manual():
    """Creates a multi-page PDF document on Aircraft Maintenance."""
    pdf_path = SAMPLE_DIR / "Aircraft_Maintenance_Manual_B737.pdf"
    doc = fitz.open()

    # Page 1: Daily and Pre-Flight Checks
    page1 = doc.new_page(width=595, height=842)  # A4 size
    text_page1 = """AIRCRAFT MAINTENANCE MANUAL (AMM) - BOEING 737 SERIES
DOCUMENT REF: AMM-B737-REV42
SECTION 1: DAILY INSPECTIONS & PRE-FLIGHT MAINTENANCE PROCEDURES

1.1 GENERAL MAINTENANCE POLICY
All aircraft operating under institutional airworthiness regulations must undergo scheduled
and unscheduled maintenance according to the approved Maintenance Control Program. Maintenance
records must be preserved for minimum 24 months.

1.2 PRE-FLIGHT WALK-AROUND INSPECTION
Before each departure, certified maintenance personnel or the flight crew must perform an external
walk-around inspection adhering to standard checklist sequence:
- Pitot-static probes: Check covers removed, openings unobstructed by foreign objects or ice.
- Radome and forward fuselage: Inspect for bird strike impacts, hail damage, or skin wrinkles.
- Landing gear assembly:
  * Check main landing gear shock strut extension (nominal 4.5 to 5.5 inches).
  * Inspect tire condition: Minimum tread groove depth is 2.0 mm; verify absence of deep cuts.
  * Check brake wear indicator pins; replace brake assembly if pin is flush with housing.
  * Main tire pressure must be maintained at 205 ± 5 PSI at cold ambient temperatures.
- Hydraulic systems: Verify Reservoirs A, B, and Standby are pressurized above 2,800 PSI.
- Engine cowlings and fan blades: Visually inspect all 24 titanium fan blades for nicks, cracks,
  or leading-edge deformation. Minimum allowable oil quantity before dispatch is 12 quarts.
"""
    page1.insert_text((50, 60), text_page1, fontsize=10, fontname="helv")

    # Page 2: Scheduled Maintenance Check Intervals
    page2 = doc.new_page(width=595, height=842)
    text_page2 = """AIRCRAFT MAINTENANCE MANUAL (AMM) - BOEING 737 SERIES
SECTION 2: SCHEDULED MAINTENANCE INTERVALS (A, B, C, AND D CHECKS)

2.1 MAINTENANCE INTERVAL CLASSIFICATIONS
To maintain airworthiness certification, maintenance is grouped into progressive check cycles:

- A-CHECK (Every 400 to 600 Flight Hours or 60 Days):
  Routine visual inspection performed in hangar overnight. Includes lubrication of flight control
  surfaces, inspection of emergency evacuation slide inflation cylinders, checking engine fuel filters,
  and testing APU (Auxiliary Power Unit) fire detection loops.

- B-CHECK (Every 6 to 8 Months):
  Detailed mechanical alignment check. Calibration of flight instruments, electrical bus isolation
  testing, and inspection of wheel well wiring harnesses.

- C-CHECK (Every 20 to 24 Months or 6,000 Flight Hours):
  Comprehensive structural and systems overhaul requiring aircraft grounding for 1 to 2 weeks.
  Involves non-destructive ultrasonic testing (NDT) of wing spar attach fittings, flap track fairing
  actuators, horizontal stabilizer jack screw lubrication, and complete cabin pressurization leak test.

- D-CHECK (Every 6 to 10 Years):
  Heavy maintenance visit (HMV). The aircraft is entirely stripped of paint, interior cabin removed,
  engines dismounted, and the fuselage undergo complete metallurgical fatigue testing.
"""
    page2.insert_text((50, 60), text_page2, fontsize=10, fontname="helv")

    # Page 3: Emergency Landing Gear and Hydraulic Failure Procedures
    page3 = doc.new_page(width=595, height=842)
    text_page3 = """AIRCRAFT MAINTENANCE MANUAL (AMM) - BOEING 737 SERIES
SECTION 3: HYDRAULIC ANOMALIES & MANUAL GEAR EXTENSION MAINTENANCE

3.1 HYDRAULIC SYSTEM REDUNDANCY
The aircraft operates with two independent primary hydraulic systems (System A and System B)
and an auxiliary Standby System. Nominal operating pressure is 3,000 PSI (+/- 100 PSI).

3.2 MANUAL GEAR EXTENSION MECHANISM
In the event of total dual hydraulic pressure loss (System A and System B failure), the landing gear
must be deployed via the manual extension handles located on the flight deck floor behind the first officer.
- Manual extension relies on gravitational free-fall and aerodynamic air load locking.
- Maintenance Inspection Protocol for Manual Extension Cables:
  1. Inspect stainless steel tension cables for frays, kinks, or corrosion every C-Check.
  2. Cable tension must be calibrated to 45 lbs (+/- 3 lbs) at 20 deg C using calibrated tensiometer.
  3. Uplock mechanical release cams must be lubricated using AeroShell Grease 22.
  4. Perform functional gear free-fall swing test on hangar jacks following any component replacement.
"""
    page3.insert_text((50, 60), text_page3, fontsize=10, fontname="helv")

    doc.save(str(pdf_path))
    doc.close()
    print(f"Created PDF: {pdf_path}")


def create_docx_safety_guidelines():
    """Creates a DOCX document on Flight Operations Safety."""
    docx_path = SAMPLE_DIR / "Flight_Operations_Safety_Guidelines.docx"
    doc = docx.Document()

    doc.add_heading("FLIGHT OPERATIONS SAFETY GUIDELINES", level=0)
    doc.add_paragraph("Institutional Flight Operations Manual - Volume IV (Operational Safety Standards)")
    doc.add_paragraph("Document ID: FOM-SAF-2024-V4 | Distribution: Flight Crew & Ground Operations")

    doc.add_heading("1. Pre-Flight Safety Briefing Procedures", level=1)
    doc.add_paragraph(
        "Before every operational flight sector, the Pilot-in-Command (PIC) must convene a mandatory "
        "joint crew safety briefing. The briefing must cover: forecasted en-route weather anomalies, "
        "turbulence SIGMETs, aircraft Minimum Equipment List (MEL) operational caveats, critical terrain "
        "escape routes, and specific sterile flight deck expectations."
    )
    doc.add_paragraph(
        "Cabin crew briefings must confirm cabin emergency equipment readiness, designated able-bodied "
        "passengers (ABPs) adjacent to emergency exit rows, and verification of slide arming protocols."
    )

    doc.add_heading("2. Sterile Cockpit Safety Rule", level=1)
    doc.add_paragraph(
        "The sterile flight deck rule is strictly enforced during all critical phases of flight below "
        "10,000 feet Mean Sea Level (MSL), as well as during taxi, takeoff, and landing operations. "
        "During this period, all non-essential communication, paperwork, or passenger announcements "
        "are prohibited. Only safety-critical alerts concerning flight path, aircraft systems integrity, "
        "or runway hazards may be voiced."
    )

    doc.add_heading("3. Bird Strike Avoidance and Action Protocol", level=1)
    doc.add_paragraph(
        "Flock activity and wildlife presence must be monitored via ATIS and airport wildlife advisories. "
        "If bird flock presence is reported on departure corridor:"
    )
    doc.add_paragraph("- Maximize climb thrust to clear flock altitude rapidly.")
    doc.add_paragraph("- Turn on all aircraft exterior lights (landing, taxi, logo lights) to enhance aircraft visibility to avian species.")
    doc.add_paragraph(
        "If a bird strike occurs during takeoff after V1 (takeoff decision speed):"
    )
    doc.add_paragraph(
        "Continue takeoff run and establish positive rate of climb. Do NOT make abrupt throttle reductions. "
        "Once established at minimum safe altitude (MSA), cross-check engine vibration indicators, EGT (exhaust gas temp), "
        "and N1/N2 rotor speeds. If engine fire or uncontained failure is confirmed, execute Engine Fire Checklist."
    )

    doc.add_heading("4. Cabin Depressurization Safety Protocols", level=1)
    doc.add_paragraph(
        "In the event of rapid cabin depressurization (cabin altitude climbing past 10,000 feet):"
    )
    doc.add_paragraph("1. Don oxygen masks immediately and set regulator to 100% Emergency oxygen.")
    doc.add_paragraph("2. Establish flight deck crew communications via interphone.")
    doc.add_paragraph("3. Disengage autothrottle, extend speedbrakes, and initiate Emergency Descent.")
    doc.add_paragraph("4. Target descent to 10,000 feet MSL or lowest Minimum Off-Route Altitude (MORA).")
    doc.add_paragraph("5. Passenger emergency oxygen masks deploy automatically when cabin altitude exceeds 14,000 feet.")

    doc.save(str(docx_path))
    print(f"Created DOCX: {docx_path}")


def create_txt_runway_and_emergency():
    """Creates a TXT document on Runway Safety and Emergency Response."""
    txt_path = SAMPLE_DIR / "Emergency_Response_and_Runway_Safety.txt"
    content = """INSTITUTIONAL AVIATION SAFETY PROTOCOL
DOCUMENT: AIRPORT OPERATIONS & RUNWAY SAFETY MANUAL (AORS-MAN-09)
TOPIC: RUNWAY INCURSION PREVENTION, RUNWAY SAFETY, AND EMERGENCY RESPONSE

======================================================================
CHAPTER 1: RUNWAY SAFETY & INCURSION MITIGATION PROCEDURES
======================================================================

1.1 DEFINITION OF RUNWAY INCURSION
A runway incursion is any occurrence at an aerodrome involving the incorrect presence of an aircraft,
vehicle, or person on the protected area of a surface designated for the landing and takeoff of aircraft.

1.2 HOLD SHORT COMPLIANCE
- Pilots and ground vehicle drivers must never cross a solid hold short line without an explicit ATC clearance.
- An ATC instruction such as "Taxi to Runway 28R" is NOT a clearance to cross or enter Runway 28R.
- Flight crews must explicitly read back all runway hold short instructions, including runway designation.
- Both pilots must verify visual runway clearance (left and right) before crossing any active runway threshold.

1.3 LOW VISIBILITY RUNWAY SAFETY PROCEDURES (LVP)
When Runway Visual Range (RVR) drops below 600 meters or ceiling drops below 200 feet:
- Low Visibility Operations (LVO) are declared by Air Traffic Control.
- Surface movement radar (ASDE-X) tracking is mandated for all movements.
- Aircraft must maintain standard taxi speeds not exceeding 10 knots on curved taxiways.
- Intersection takeoffs are strictly prohibited during LVO; full runway length departures are required.
- Runway stop bar lights must remain illuminated; crossing an illuminated red stop bar is an acute safety violation.

======================================================================
CHAPTER 2: EMERGENCY PROCEDURES - REJECTED TAKEOFF (RTO)
======================================================================

2.1 REJECTED TAKEOFF DECISION CRITERIA
The decision to reject a takeoff must be made before reaching V1 (takeoff decision speed):
- Below 80 knots: Reject takeoff for any master caution, system failure, engine anomaly, or unusual noise.
- Between 80 knots and V1: Reject ONLY for fire warning, engine failure, catastrophic windshear alert,
  or an unsafe condition that renders the aircraft unflyable.
- Above V1: The aircraft MUST continue the takeoff. Rejecting above V1 risks catastrophic runway overrun.

2.2 RTO EXECUTION SEQUENCE
1. Captain simultaneously closes thrust levers and applies maximum manual wheel braking (or verifies RTO autobrake engagement).
2. Deploy reverse thrust to maximum allowable level.
3. Deploy speedbrakes/ground spoilers immediately.
4. Stop aircraft on runway, set parking brake, and notify ATC with "MAYDAY, MAYDAY, MAYDAY".
5. Direct fire rescue teams to inspect wheel brakes for thermal fuse plug deflation or tire fires.

======================================================================
CHAPTER 3: AIRPORT RESCUE AND FIREFIGHTING (ARFF) COORDINATION
======================================================================

3.1 EMERGENCY RESPONSE CLASSIFICATIONS
- Alert 1 (Local Standby): An approaching aircraft has known minor defect; response units position on standby.
- Alert 2 (Full Emergency): An aircraft has major system impairment (e.g., engine fire, hydraulic loss, gear failure);
  all ARFF crash tenders deploy to designated runway staging areas.
- Alert 3 (Aircraft Accident): An aircraft accident has occurred on or adjacent to airport property;
  immediate mutual aid response and emergency triage activated.
"""
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created TXT: {txt_path}")


if __name__ == "__main__":
    create_pdf_maintenance_manual()
    create_docx_safety_guidelines()
    create_txt_runway_and_emergency()
    print("All sample documents successfully generated.")
