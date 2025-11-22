import os
import re
import time
import random
import json
import requests
import threading
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor

try:
    from playwright_stealth import stealth_sync
    STEALTH_AVAILABLE = True
except Exception:
    STEALTH_AVAILABLE = False

try:
    from fake_useragent import UserAgent
except Exception:
    UserAgent = None

# Gemini SDK
import google.generativeai as genai

# Flask
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# -------------------------------------------------
# 🔥 CONFIG FOR RENDER (ENV VARIABLES)
# -------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT", 5432)
}

PROXIES = []

# -------------------------------------------------
# Flask
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

scraping_status = {
    'is_running': False,
    'current_job': None,
    'progress': 0,
    'total_jobs': 0,
    'start_time': None
}

# -------------------------------------------------
# Gemini Init
# -------------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)

try:
    gemini_model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print("Gemini init error:", e)
    gemini_model = None

# -------------------------------------------------
# Database
# -------------------------------------------------
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def init_database():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS indeed_jobs (
            id SERIAL PRIMARY KEY,
            title TEXT,
            company TEXT,
            location TEXT,
            posted_time TEXT,
            description TEXT,
            employment_type TEXT,
            seniority_level TEXT,
            industries TEXT,
            job_function TEXT,
            company_url TEXT,
            company_logo TEXT,
            company_about TEXT,
            job_url TEXT,
            extracted_skills TEXT,
            salary_range TEXT,
            applicant_count TEXT,
            source_platform TEXT,
            scraped_at TEXT,
            search_keywords TEXT,
            search_location TEXT
        );
        """

        cur.execute(create_table_sql)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Database initialized")
    except Exception as e:
        print("❌ Database init error:", e)


# -------------------------------------------------
# UTILITIES  (same as your code — unchanged)
# -------------------------------------------------
def rand_sleep(a=0.6, b=1.6): time.sleep(random.uniform(a, b))
def human_scroll(page, steps=6):
    for _ in range(steps):
        try: page.mouse.wheel(0, random.randint(300, 900))
        except: pass
        rand_sleep(0.6, 1.2)

def random_viewport(): return {
    "width": random.choice([1200, 1280, 1366, 1400]),
    "height": random.choice([700, 768, 800, 900])
}

def random_user_agent():
    if UserAgent:
        try: return UserAgent().random
        except: pass
    return "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

STOP_WORDS = set([
    "the", "and", "for", "with", "that", "this", "from", "have", "will", "you", "your",
    "are", "our", "they", "their", "all", "any", "not", "but", "use", "used", "using",
    "required", "responsibilities", "responsibility", "role", "experience", "years",
    "work", "workplace", "team", "ability", "skills", "including", "including:"
])

def extract_skills_dynamic(text, max_skills=30):
    if not text or len(text) < 40:
        return []
    text_clean = re.sub(r'\s+', ' ', text)
    token_pattern = re.compile(r'\b[A-Za-z0-9\+\#\.\-]{2,40}\b')
    raw_tokens = token_pattern.findall(text_clean)
    candidates = []
    for tok in raw_tokens:
        if tok.lower() in STOP_WORDS:
            continue
        if re.fullmatch(r'\d+', tok):
            continue
        if (re.search(r'[A-Z][a-z]+[A-Z][a-z]+', tok) or
            re.search(r'[\+\#\.]', tok) or
            tok.isupper() or
            re.search(r'[A-Za-z]{2,}\d+', tok) or
            (len(tok) <= 20 and tok[0].isalpha() and tok[0].isupper())):
            candidates.append(tok.strip('.,:;()[]'))
    phrase_pattern = re.compile(r'\b([A-Za-z0-9\+\#\.\-]{2,30}(?:\s+[A-Za-z0-9\+\#\.\-]{2,30}){0,2})\b')
    for match in phrase_pattern.findall(text_clean):
        if any(tok in match for tok in candidates[:200]):
            candidates.append(match.strip('.,:;()[]'))
    normalized = []
    seen = set()
    for c in candidates:
        s = c.strip()
        s_low = s.lower()
        if s_low in seen:
            continue
        if len(s) > 100:
            continue
        if re.fullmatch(r'[^\w]*', s):
            continue
        words = [w for w in re.split(r'\s+', s) if w.lower() not in STOP_WORDS]
        if not words:
            continue
        seen.add(s_low)
        normalized.append(s)
        if len(normalized) >= max_skills:
            break
    return normalized

def gemini_extract_fields(description, max_retries=2):
    if not gemini_model or not description or len(description.strip()) < 30:
        return {}

    prompt = f"""
You are a strict JSON extractor for job postings. Given the job description below,
extract these fields and return ONLY valid JSON:

- posted_time: e.g., "1 week ago" or empty string
- employment_type: "Full-time", "Part-time", "Contract", "Internship", etc.
- seniority_level: "Entry level", "Mid-Senior level", "Senior", "Lead", etc.
- industries: comma-separated categories or empty string
- job_function: short phrase, e.g., "Engineering and Information Technology"
- company_about: a 1-2 sentence summary of the company (if inferable)
- extracted_skills: JSON array of skills mentioned or clearly implied (do not invent)
- salary_range: if present in the description
- applicant_count: if present

Return exactly JSON. Example:
{{
  "posted_time": "2 days ago",
  "employment_type": "Full-time",
  "seniority_level": "Mid-Senior level",
  "industries": "Technology, Information and Internet",
  "job_function": "Engineering and Information Technology",
  "company_about": "A one-two sentence summary...",
  "extracted_skills": ["Python","TensorFlow"],
  "salary_range": "₹5,00,000 - ₹8,00,000 a year",
  "applicant_count": "Over 200 applicants"
}}

Job Description:

{description}

"""

    for attempt in range(1, max_retries + 1):
        try:
            resp = gemini_model.generate_content(prompt)
            raw = resp.text.strip()
            match = re.search(r'(\{[\s\S]*\})', raw)
            raw_json = match.group(1) if match else raw
            data = json.loads(raw_json)
            if isinstance(data.get("extracted_skills"), str):
                data["extracted_skills"] = [s.strip() for s in data["extracted_skills"].split(",") if s.strip()]
            if isinstance(data.get("industries"), list):
                data["industries"] = ", ".join(data["industries"])
            return data
        except Exception as e:
            print(f"[Gemini attempt {attempt}] parse error: {e}")
            time.sleep(0.5 * attempt)
            continue

    fallback = {}
    m = re.search(r'(\b\d+\s+(?:day|days|week|weeks|month|months|hour|hours)\s+ago\b)', description, re.I)
    fallback["posted_time"] = m.group(1) if m else ""
    emp = re.search(r'\b(full[-\s]?time|part[-\s]?time|contract|internship|temporary)\b', description, re.I)
    fallback["employment_type"] = emp.group(1).capitalize() if emp else ""
    sv = re.search(r'\b(entry level|junior|mid[-\s]?level|mid[-\s]?senior|senior|lead|manager|director)\b', description, re.I)
    fallback["seniority_level"] = sv.group(1).title() if sv else ""
    fallback["extracted_skills"] = extract_skills_dynamic(description)
    sal = re.search(r'₹[\d,]+(?:\s*-\s*₹[\d,]+)?\s*(?:a year|per year|p\.a\.)?', description)
    fallback["salary_range"] = sal.group(0) if sal else ""
    appl = re.search(r'(\b\d{1,6}\+?\s+applicants?\b)', description, re.I)
    fallback["applicant_count"] = appl.group(1) if appl else ""
    fallback["industries"] = ""
    fallback["job_function"] = ""
    fallback["company_about"] = ""
    return fallback

def parse_viewjob_html(html, job_id=None, view_url=None, search_keywords="", search_location=""):
    soup = BeautifulSoup(html, "html.parser")
    job = {}

    desc_el = soup.select_one("#jobDescriptionText") or soup.select_one("div.JobDescription")
    if not desc_el:
        return None
    description = desc_el.get_text(" ", strip=True)
    job["description"] = description

    title = None
    for sel in [
        "h1.jobsearch-JobInfoHeader-title",
        "h1[data-testid='jobTitle']",
        "div.jobsearch-JobInfoHeader-title-container h1",
        "h1"
    ]:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            title = el.get_text(strip=True)
            break
    if not title:
        meta_title = soup.select_one("meta[property='og:title']") or soup.select_one("meta[name='title']")
        if meta_title and meta_title.get("content"):
            title = meta_title["content"].strip()
    job["title"] = title

    company = None
    for sel in [
        "[data-testid='company-name']",
        "div.jobsearch-InlineCompanyRating div:first-child",
        ".icl-u-lg-mr--sm.icl-u-xs-mr--xs",
        "div.jobsearch-CompanyInfoWithoutHeaderImage a",
        "div[data-company-name]",
        "div[data-testid='companyInfo'] span"
    ]:
        el = soup.select_one(sel)
        if el:
            txt = el.get_text(strip=True)
            if txt:
                company = txt
                break
    job["company"] = company

    location = None
    for sel in [
        "div.jobsearch-JobInfoHeader-subtitle div",
        "div.jobsearch-JobInfoHeader-subtitle span",
        "[data-testid='text-location']",
        "div[data-testid='inlineHeader-companyLocation']",
        ".jobsearch-CompanyInfoWithoutHeaderImage div"
    ]:
        el = soup.select_one(sel)
        if el:
            txt = el.get_text(strip=True)
            if txt:
                location = txt
                break
    job["location"] = location

    company_url = None
    for sel in [
        "[data-testid='company-name'] a",
        "div.jobsearch-InlineCompanyRating a",
        "div.jobsearch-CompanyInfoWithoutHeaderImage a",
        "div[data-company-name] a",
        "div[data-testid='companyInfo'] a"
    ]:
        el = soup.select_one(sel)
        if el and el.get("href"):
            href = el.get("href")
            if href.startswith("http"):
                company_url = href
            else:
                company_url = "https://in.indeed.com" + href
            break
    job["company_url"] = company_url

    footer = soup.select_one("div.jobsearch-JobMetadataFooter")
    if footer:
        job["posted_time"] = footer.get_text(" ", strip=True)
    else:
        job["posted_time"] = ""

    salary = None
    for sel in ["span.jobsearch-JobMetadataHeader-iconLabel", "div#salaryInfo", "div[data-testid='salaryInfo']"]:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            salary = el.get_text(" ", strip=True)
            break
    job["salary_range"] = salary or ""

    job["employment_type"] = ""
    job["seniority_level"] = ""
    job["industries"] = ""
    job["job_function"] = ""
    job["company_about"] = ""
    job["applicant_count"] = ""
    job["extracted_skills"] = ""

    job["job_url"] = view_url if view_url else (f"https://in.indeed.com/viewjob?jk={job_id}" if job_id else None)
    job["scraped_at"] = datetime.now().isoformat()
    job["source_platform"] = "indeed"
    job["search_keywords"] = search_keywords
    job["search_location"] = search_location
    job["company_logo"] = None

    return job

def get_viewjob_html(job_id, page=None, proxy=None):
    view_url = f"https://in.indeed.com/viewjob?jk={job_id}"
    headers = {
        "User-Agent": random_user_agent(),
        "Accept-Language": "en-US,en;q=0.9"
    }
    try:
        resp = requests.get(view_url, headers=headers, timeout=15, proxies={"http": proxy, "https": proxy} if proxy else None)
        if resp.status_code == 200:
            text = resp.text
            if "#jobDescriptionText" in text or "jobDescriptionText" in text:
                return text, view_url
    except Exception as e:
        print("requests fallback error:", e)

    if page:
        try:
            page.goto(view_url, timeout=30000, wait_until="domcontentloaded")
            rand_sleep(1.0, 2.2)
            try:
                human_scroll(page, steps=random.randint(3, 7))
            except Exception:
                pass
            page.wait_for_timeout(random.randint(400, 900))
            html = page.content()
            return html, view_url
        except Exception as e:
            print("playwright get_viewjob_html error:", e)

    return None, view_url

def collect_job_ids(target_count, page, search_keywords, search_location, max_pages=50):
    job_ids = []
    start = 0
    pages_visited = 0

    while len(job_ids) < target_count and pages_visited < max_pages:
        url = f"https://in.indeed.com/jobs?q={requests.utils.requote_uri(search_keywords)}&l={requests.utils.requote_uri(search_location)}&start={start}"
        print("Visiting listing:", url)
        try:
            page.goto(url, timeout=45000, wait_until="domcontentloaded")
        except PlaywrightTimeoutError:
            print("Timeout on listing page")
        rand_sleep(1.0, 2.0)
        human_scroll(page, steps=random.randint(3, 6))
        rand_sleep(0.6, 1.2)

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")
        page_text = soup.get_text(" ", strip=True).lower()
        if "human verification" in page_text or "verify you are a human" in page_text or "captcha" in page_text or "please verify" in page_text:
            print("CAPTCHA/verification detected on listing page. Abort collection.")
            break

        cards = soup.select("div.job_seen_beacon")
        if not cards:
            print("No job cards found - possible block or layout change.")
            break

        for c in cards:
            if len(job_ids) >= target_count:
                break
            jk = None
            a = c.select_one("a")
            if a and a.get("data-jk"):
                jk = a.get("data-jk")
            elif a and "jk=" in (a.get("href") or ""):
                part = (a.get("href") or "").split("jk=")[-1]
                jk = part.split("&")[0].strip()
            elif c.get("data-jk"):
                jk = c.get("data-jk")
            if jk and jk not in job_ids:
                job_ids.append(jk)
        start += 10
        pages_visited += 1
        rand_sleep(1.0, 2.0)

    print(f"Collected {len(job_ids)} job ids")
    return job_ids

def save_to_postgres(results):
    conn = get_db_connection()
    cur = conn.cursor()

    insert_sql = """
    INSERT INTO indeed_jobs (
        title, company, location, posted_time, description, employment_type,
        seniority_level, industries, job_function, company_url, company_logo,
        company_about, job_url, extracted_skills, salary_range, applicant_count,
        source_platform, scraped_at, search_keywords, search_location
    ) VALUES %s
    """

    rows = []
    for job in results:
        rows.append((
            job.get("title"),
            job.get("company"),
            job.get("location"),
            job.get("posted_time"),
            job.get("description"),
            job.get("employment_type"),
            job.get("seniority_level"),
            job.get("industries"),
            job.get("job_function"),
            job.get("company_url"),
            job.get("company_logo"),
            job.get("company_about"),
            job.get("job_url"),
            job.get("extracted_skills"),
            job.get("salary_range"),
            job.get("applicant_count"),
            job.get("source_platform"),
            job.get("scraped_at"),
            job.get("search_keywords"),
            job.get("search_location")
        ))

    execute_values(cur, insert_sql, rows)
    conn.commit()

    cur.close()
    conn.close()
    print(f"\nSaved {len(rows)} records into PostgreSQL successfully!")

def scrape(search_keywords, search_location, target_count, use_proxies=False):
    results = []
    proxy_index = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            '--single-process'
        ])

        ua = random_user_agent()
        viewport = random_viewport()
        context = browser.new_context(user_agent=ua, viewport=viewport, locale="en-US")
        page = context.new_page()

        if STEALTH_AVAILABLE:
            try:
                stealth_sync(page)
            except Exception:
                pass

        job_ids = collect_job_ids(target_count, page, search_keywords, search_location)
        if not job_ids:
            print("Retrying collection with new context...")
            context.close()
            ua = random_user_agent()
            viewport = random_viewport()
            context = browser.new_context(user_agent=ua, viewport=viewport, locale="en-US")
            page = context.new_page()
            if STEALTH_AVAILABLE:
                try:
                    stealth_sync(page)
                except Exception:
                    pass
            job_ids = collect_job_ids(target_count, page, search_keywords, search_location)

        if not job_ids:
            print("Failed to collect job ids (blocked by verification). Exiting.")
            browser.close()
            return

        proxies = PROXIES if use_proxies else []

        for idx, job_id in enumerate(job_ids, start=1):
            if len(results) >= target_count:
                break
            print(f"[{idx}/{len(job_ids)}] Processing job id: {job_id}")

            # Update scraping status
            global scraping_status
            scraping_status['progress'] = idx
            scraping_status['total_jobs'] = len(job_ids)

            proxy = proxies[proxy_index % len(proxies)] if proxies else None
            proxy_index += 1

            html, view_url = get_viewjob_html(job_id, page=page, proxy=proxy)
            if not html:
                print("Failed to fetch viewjob html for", job_id)
                continue

            if "verify" in html.lower() or "human verification" in html.lower() or "captcha" in html.lower():
                print("Verification page detected when loading job. Trying context reset and retry...")
                try:
                    context.close()
                except Exception:
                    pass
                ua = random_user_agent()
                viewport = random_viewport()
                context = browser.new_context(user_agent=ua, viewport=viewport, locale="en-US")
                page = context.new_page()
                if STEALTH_AVAILABLE:
                    try:
                        stealth_sync(page)
                    except Exception:
                        pass
                html, view_url = get_viewjob_html(job_id, page=page, proxy=proxy)
                if not html or "verify" in html.lower():
                    print("Still blocked for job:", job_id, "-> skipping.")
                    continue

            parsed = parse_viewjob_html(html, job_id=job_id, view_url=view_url, search_keywords=search_keywords, search_location=search_location)
            if not parsed:
                print("Skipping (no description) for", job_id)
                continue

            ai = gemini_extract_fields(parsed["description"])
            parsed["posted_time"] = ai.get("posted_time") or parsed.get("posted_time") or ""
            parsed["employment_type"] = ai.get("employment_type") or parsed.get("employment_type") or ""
            parsed["seniority_level"] = ai.get("seniority_level") or parsed.get("seniority_level") or ""
            parsed["industries"] = ai.get("industries") or parsed.get("industries") or ""
            parsed["job_function"] = ai.get("job_function") or parsed.get("job_function") or ""
            parsed["company_about"] = ai.get("company_about") or parsed.get("company_about") or ""
            skills = ai.get("extracted_skills")
            if isinstance(skills, list):
                parsed["extracted_skills"] = ", ".join(skills)
            elif isinstance(skills, str) and skills.strip():
                parsed["extracted_skills"] = skills
            else:
                parsed["extracted_skills"] = parsed.get("extracted_skills") or ""
            parsed["salary_range"] = ai.get("salary_range") or parsed.get("salary_range") or ""
            parsed["applicant_count"] = ai.get("applicant_count") or parsed.get("applicant_count") or ""
            parsed["company_logo"] = parsed.get("company_logo") or None

            results.append(parsed)
            print("Collected:", parsed.get("title") or job_id)
            rand_sleep(0.5, 1.6)

        try:
            context.close()
        except Exception:
            pass
        browser.close()

    # Save to database
    if results:
        save_to_postgres(results)

    return results


# -------------------------
# Flask API Routes
# -------------------------
@app.route('/')
def home():
    """Serve the web interface"""
    return render_template('index.html')

@app.route('/api/jobs', methods=['GET'])
def get_all_jobs():
    """Get all jobs from the database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("SELECT * FROM indeed_jobs ORDER BY scraped_at DESC")
        jobs = cur.fetchall()

        jobs_list = []
        for job in jobs:
            job_dict = dict(job)
            for key, value in job_dict.items():
                if value is None:
                    job_dict[key] = ""
            jobs_list.append(job_dict)

        cur.close()
        conn.close()

        return jsonify({
            'success': True,
            'count': len(jobs_list),
            'jobs': jobs_list
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/jobs/search', methods=['GET'])
def search_jobs():
    """Search jobs by keywords, location, or company"""
    try:
        keyword = request.args.get('keyword', '')
        location = request.args.get('location', '')
        company = request.args.get('company', '')

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        query = "SELECT * FROM indeed_jobs WHERE 1=1"
        params = []

        if keyword:
            query += " AND (title ILIKE %s OR description ILIKE %s OR search_keywords ILIKE %s)"
            params.extend([f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'])

        if location:
            query += " AND (location ILIKE %s OR search_location ILIKE %s)"
            params.extend([f'%{location}%', f'%{location}%'])

        if company:
            query += " AND company ILIKE %s"
            params.append(f'%{company}%')

        query += " ORDER BY scraped_at DESC"

        cur.execute(query, params)
        jobs = cur.fetchall()

        jobs_list = []
        for job in jobs:
            job_dict = dict(job)
            for key, value in job_dict.items():
                if value is None:
                    job_dict[key] = ""
            jobs_list.append(job_dict)

        cur.close()
        conn.close()

        return jsonify({
            'success': True,
            'count': len(jobs_list),
            'jobs': jobs_list
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/scrape', methods=['POST'])
def start_scraping():
    """Start a new scraping job"""
    global scraping_status

    if scraping_status['is_running']:
        return jsonify({
            'success': False,
            'error': 'Scraping is already in progress'
        }), 409

    data = request.json
    job_title = data.get('job_title', '').strip()
    location = data.get('location', '').strip()
    num_jobs = data.get('num_jobs', 10)

    if not job_title:
        return jsonify({
            'success': False,
            'error': 'Job title is required'
        }), 400

    try:
        num_jobs = int(num_jobs)
        if num_jobs <= 0 or num_jobs > 50:
            return jsonify({
                'success': False,
                'error': 'Number of jobs must be between 1 and 50'
            }), 400
    except ValueError:
        return jsonify({
            'success': False,
            'error': 'Number of jobs must be a valid integer'
        }), 400

    scraping_status = {
        'is_running': True,
        'current_job': f"{job_title} in {location}",
        'progress': 0,
        'total_jobs': num_jobs,
        'start_time': datetime.now().isoformat()
    }

    thread = threading.Thread(
        target=run_scraping,
        args=(job_title, location, num_jobs)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'success': True,
        'message': f'Scraping started for {job_title} in {location}',
        'scraping_id': int(time.time())
    })

def run_scraping(job_title, location, num_jobs):
    """Run the scraping process in background"""
    global scraping_status

    try:
        scrape(
            search_keywords=job_title,
            search_location=location,
            target_count=num_jobs,
            use_proxies=False
        )

        scraping_status = {
            'is_running': False,
            'current_job': None,
            'progress': 100,
            'total_jobs': num_jobs,
            'start_time': scraping_status['start_time'],
            'end_time': datetime.now().isoformat()
        }

    except Exception as e:
        scraping_status = {
            'is_running': False,
            'current_job': None,
            'progress': 0,
            'total_jobs': num_jobs,
            'start_time': scraping_status['start_time'],
            'end_time': datetime.now().isoformat(),
            'error': str(e)
        }

@app.route('/api/scraping/status', methods=['GET'])
def get_scraping_status():
    """Get current scraping status"""
    return jsonify(scraping_status)

@app.route('/api/jobs/stats', methods=['GET'])
def get_job_stats():
    """Get statistics about stored jobs"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM indeed_jobs")
        total_jobs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT company) FROM indeed_jobs WHERE company IS NOT NULL AND company != ''")
        total_companies = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT location) FROM indeed_jobs WHERE location IS NOT NULL AND location != ''")
        total_locations = cur.fetchone()[0]

        cur.execute("SELECT MAX(scraped_at) FROM indeed_jobs")
        last_scraped = cur.fetchone()[0]
    
        cur.close()
        conn.close()

        return jsonify({
            'success': True,
            'stats': {
                'total_jobs': total_jobs,
                'total_companies': total_companies,
                'total_locations': total_locations,
                'last_scraped': last_scraped
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/jobs/<int:job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a specific job by ID"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("DELETE FROM indeed_jobs WHERE id = %s", (job_id,))
        conn.commit()

        if cur.rowcount == 0:
            cur.close()
            conn.close()
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404

        cur.close()
        conn.close()

        return jsonify({
            'success': True,
            'message': 'Job deleted successfully'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/jobs/clear', methods=['DELETE'])
def clear_all_jobs():
    """Clear all jobs from database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("DELETE FROM indeed_jobs")
        conn.commit()

        cur.close()
        conn.close()

        return jsonify({
            'success': True,
            'message': 'All jobs cleared successfully'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@app.route('/db-test')
def db_test():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        return "DB OK", 200
    except Exception as e:
        return str(e), 500
# -------------------------
# Main Execution
if __name__ == "__main__":
    print("🚀 Initializing DB...")
    init_database()

    # Render provides PORT env variable
    port = int(os.getenv("PORT", 10000))

    print(f"🔥 Starting Flask on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

