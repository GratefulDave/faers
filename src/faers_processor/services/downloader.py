"""Service for downloading FAERS data."""
import concurrent.futures
import logging
import os
import re
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Set

import aiofiles
import aiohttp
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class DataDownloader(ABC):
    """Abstract base class for data downloading."""

    @abstractmethod
    async def download_file(self, url: str, destination: Path) -> None:
        """Download a single file."""
        pass

    @abstractmethod
    async def extract_file(self, zip_path: Path, extract_path: Path) -> None:
        """Extract a downloaded zip file."""
        pass


class FAERSDownloader(DataDownloader):
    """FAERS specific data downloader implementation."""
    
    FAERS_BASE_URL = "https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html"
    FAERS_CONTENT_BASE = "https://fis.fda.gov"
    FAERS_START_YEAR = 2004  # Earliest available data
    FAERS_SWITCH_YEAR = 2013  # Year when AERS switched to FAERS
    
    def __init__(self, output_dir: Path):
        """Initialize the FAERS downloader.
        
        Args:
            output_dir: Base directory for downloaded files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = None
        self.quarter_urls = {}  # Cache for quarter URLs
        
    @staticmethod
    def get_prefix_for_year(year: int) -> str:
        """Get the correct prefix (aers/faers) for a given year."""
        return 'faers' if year >= FAERSDownloader.FAERS_SWITCH_YEAR else 'aers'
        
    @staticmethod
    def validate_quarter(quarter: str) -> bool:
        """Validate quarter format and range.
        
        Args:
            quarter: Quarter string (e.g., '2023q1')
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If quarter format or range is invalid
        """
        try:
            if not re.match(r'^\d{4}q[1-4]$', quarter.lower()):
                raise ValueError(f"Invalid quarter format: {quarter}")
                
            year = int(quarter[:4])
            if year < FAERSDownloader.FAERS_START_YEAR:
                raise ValueError(f"Year {year} predates available FAERS data")
                
            return True
        except Exception as e:
            raise ValueError(f"Invalid quarter {quarter}: {str(e)}")
            
    def get_quarters(self) -> List[str]:
        """Get list of available FAERS quarters from FDA website.
        
        Returns:
            List of quarter identifiers (e.g., ['2023q1', '2023q2'])
        """
        try:
            # Get the webpage content
            response = requests.get(self.FAERS_BASE_URL, timeout=30)
            response.raise_for_status()

            # Parse HTML and find zip file links
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Dictionary to store quarter URLs
            quarter_urls = {}
            
            # Process both current and older files sections
            for section in ['fpd-panel', 'fpd-panel-legacy']:
                panels = soup.find_all('div', class_=section)
                for panel in panels:
                    # Find all ASCII zip file links
                    links = panel.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        # Case insensitive check for zip and ascii
                        if not re.search(r'\.zip$', href, re.IGNORECASE):
                            continue
                            
                        if not re.search(r'ascii', href, re.IGNORECASE):
                            continue
                            
                        # Extract quarter from URL with flexible pattern
                        # Handle:
                        # - faers_ascii_YYYY[Qq][1-4].zip (2013-present)
                        # - aers_ascii_YYYY[Qq][1-4].zip (pre-2013)
                        match = re.search(r'(?:faers|aers)_ascii_(\d{4})([Qq])([1-4])\.zip', href, re.IGNORECASE)
                        if match:
                            year = match.group(1)
                            quarter = f"{year}q{match.group(3)}"  # Normalize to lowercase q
                            quarter_urls[quarter] = href
                            logging.debug(f"Found {quarter}: {href}")
                            
            if not quarter_urls:
                logging.error("No quarters found in webpage")
                # Log all links for debugging
                all_links = soup.find_all('a', href=True)
                logging.debug("All links found:")
                for link in all_links:
                    logging.debug(f"  {link['href']}")
                return []
                
            # Sort quarters chronologically
            sorted_quarters = sorted(quarter_urls.keys(), key=lambda x: (x[:4], x[5]))  # Sort by year then quarter number
            logging.info(f"Found {len(sorted_quarters)} quarters: {', '.join(sorted_quarters)}")
            
            # Store URLs for later use
            self.quarter_urls = quarter_urls
            
            return sorted_quarters

        except Exception as e:
            logging.error(f"Error getting quarters list: {str(e)}")
            return []
            
    def get_quarter_url(self, quarter: str) -> str:
        """Get download URL for a specific quarter."""
        if not hasattr(self, 'quarter_urls'):
            self.get_quarters()  # Populate quarter_urls if not already done
            
        if quarter not in self.quarter_urls:
            raise ValueError(f"No URL found for quarter {quarter}")
            
        url = self.quarter_urls[quarter]
        
        # Handle relative URLs
        if not url.startswith('http'):
            # All download URLs are under the content directory
            url = f"{self.FAERS_CONTENT_BASE}/{url.lstrip('/')}"
            
        logging.debug(f"Download URL for {quarter}: {url}")
        return url

    def download_quarter(self, quarter: str) -> None:
        """Download and extract a specific FAERS quarter.
        
        Args:
            quarter: Quarter identifier (e.g., '2023q1')
        """
        try:
            self.validate_quarter(quarter)
            
            # Create quarter directory
            quarter_dir = self.output_dir / quarter
            quarter_dir.mkdir(exist_ok=True)

            # Get the direct download URL for this quarter
            download_url = self.get_quarter_url(quarter)
            
            # Determine file paths
            # Use the same prefix (faers/aers) as the source URL
            year = int(quarter[:4])
            prefix = self.get_prefix_for_year(year)
            zip_file = quarter_dir / f"{prefix}_ascii_{quarter}.zip"
            extract_dir = quarter_dir / "ascii"
            
            logging.info(f"Downloading {quarter} from {download_url}")

            # Download the zip file
            response = requests.get(download_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192

            # Only download if file doesn't exist or is incomplete
            if not zip_file.exists() or zip_file.stat().st_size != total_size:
                logging.info(f"Downloading {quarter} to {zip_file}")
                with open(zip_file, 'wb') as f:
                    with tqdm(
                            total=total_size,
                            unit='iB',
                            unit_scale=True,
                            desc=f"Downloading {quarter}"
                    ) as pbar:
                        for data in response.iter_content(block_size):
                            size = f.write(data)
                            pbar.update(size)
            else:
                logging.info(f"Zip file for {quarter} already exists and is complete")

            # Check if files are already extracted
            if extract_dir.exists() and any(extract_dir.iterdir()):
                logging.info(f"Files for {quarter} already extracted in {extract_dir}")
                return

            # Extract files
            logging.info(f"Extracting {zip_file} to {quarter_dir}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(quarter_dir)

            # Only delete zip file if extraction was successful
            if extract_dir.exists() and any(extract_dir.iterdir()):
                logging.info(f"Extraction successful, removing {zip_file}")
                zip_file.unlink()
            else:
                logging.error(f"Extraction may have failed - keeping {zip_file}")

            # Handle special case for 2018Q1
            demo_file = quarter_dir / 'ascii' / 'DEMO18Q1_new.txt'
            if demo_file.exists():
                demo_file.rename(demo_file.parent / 'DEMO18Q1.txt')

        except Exception as e:
            logging.error(f"Error downloading/extracting {quarter}: {str(e)}")
            # Don't delete zip file on error
            raise

    async def __aenter__(self):
        """Set up async session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def get_download_links(self) -> Set[str]:
        """Get all FAERS download links."""
        async with self.session.get(self.FAERS_BASE_URL) as response:
            soup = BeautifulSoup(await response.text(), 'lxml')
            links = {a['href'] for a in soup.find_all('a', href=True)
                     if '.zip' in a['href'] and 'ascii' in a['href']}
            return links

    async def download_file(self, url: str, destination: Path) -> None:
        """Download a single file asynchronously.
        
        Args:
            url: URL to download from
            destination: Path to save file to
        """
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))

                with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                    async with aiofiles.open(destination, mode='wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                            pbar.update(len(chunk))

        except Exception as e:
            logging.error(f"Error downloading {url}: {str(e)}")
            raise

    async def extract_file(self, zip_path: Path, extract_path: Path) -> None:
        """Extract a downloaded zip file.
        
        Args:
            zip_path: Path to zip file
            extract_path: Directory to extract to
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                total_size = sum(info.file_size for info in zip_ref.filelist)
                extracted_size = 0

                with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                    for info in zip_ref.filelist:
                        zip_ref.extract(info, extract_path)
                        extracted_size += info.file_size
                        pbar.update(info.file_size)

        except Exception as e:
            logging.error(f"Error extracting {zip_path}: {str(e)}")
            raise

    async def download_all(self):
        """Download and extract all FAERS data."""
        links = await self.get_download_links()
        for link in links:
            zip_name = self.output_dir / Path(link).name
            extract_dir = self.output_dir / zip_name.stem

            await self.download_file(link, zip_name)
            await self.extract_file(zip_name, extract_dir)

            # Handle special case for 2018Q1
            if '2018q1' in str(extract_dir):
                demo_file = extract_dir / 'ascii' / 'DEMO18Q1_new.txt'
                if demo_file.exists():
                    demo_file.rename(demo_file.parent / 'DEMO18Q1.txt')

    def download_all_quarters(self, max_workers: int = 4) -> None:
        """Download all available FAERS quarters.
        
        Args:
            max_workers: Maximum number of parallel downloads
        """
        quarters = self.get_quarters()
        if not quarters:
            logging.error("No quarters available for download")
            return

        logging.info(f"Found {len(quarters)} quarters available for download")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for quarter in quarters:
                future = executor.submit(
                    self.download_quarter,
                    quarter=quarter
                )
                futures.append((quarter, future))

            with tqdm(total=len(futures), desc="Downloading quarters") as pbar:
                for quarter, future in futures:
                    try:
                        future.result()
                        logging.info(f"Successfully downloaded quarter {quarter}")
                    except Exception as e:
                        logging.error(f"Error downloading quarter {quarter}: {str(e)}")
                    pbar.update(1)

    def download_all(self, max_workers: int = 4) -> None:
        """Download all available FAERS quarters.
        
        Args:
            max_workers: Maximum number of parallel downloads
        """
        return self.download_all_quarters(max_workers=max_workers)


class FAERSDataDownloader:
    """Handles downloading and extraction of FAERS data files."""

    FAERS_URL = "https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html"

    def __init__(self, raw_dir: Path, clean_dir: Path):
        """
        Initialize downloader.
        
        Args:
            raw_dir: Directory for raw FAERS data
            clean_dir: Directory for cleaned data
        """
        self.raw_dir = raw_dir
        self.clean_dir = clean_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)

    def download_latest_quarter(self) -> List[Path]:
        """
        Download the latest quarter of FAERS data.
        
        Returns:
            List of paths to downloaded files
        """
        # Get FAERS webpage
        response = requests.get(self.FAERS_URL, timeout=500)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find ASCII zip files
        zip_links = [
            link['href'] for link in soup.find_all('a')
            if link.get('href', '').endswith('.zip') and 'ascii' in link['href'].lower()
        ]

        if not zip_links:
            logging.error("No FAERS ASCII zip files found")
            return []

        # Get latest quarter
        latest_zip = zip_links[0]
        quarter = re.search(r'\d{2}q\d', latest_zip.lower()).group()

        # Download and extract
        zip_path = self.raw_dir / f"faers_{quarter}.zip"
        extract_dir = self.raw_dir / f"faers_{quarter}"

        logging.info(f"Downloading FAERS data for {quarter}")
        response = requests.get(latest_zip, stream=True)
        file_size = int(response.headers.get('content-length', 0))
        with tqdm(total=file_size, unit='iB', unit_scale=True, desc=zip_path.name) as pbar:
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Extract files
        with zipfile.ZipFile(zip_path) as zf:
            files = [f for f in zf.namelist() if f.endswith('.txt')]
            with tqdm(total=len(files), desc=f"Extracting {zip_path.name}") as pbar:
                for file in files:
                    zf.extract(file, extract_dir)
                    pbar.update(1)

        # Clean up zip file
        zip_path.unlink()

        # Fix known file issues
        self._fix_known_issues(extract_dir)

        # Return paths to extracted files
        return list(extract_dir.rglob('*.txt'))

    @staticmethod
    def _fix_known_issues(extract_dir: Path):
        """Fix known issues in FAERS files."""
        fixes = {
            'DRUG11Q2.txt': ('$$$$$$7475791', '\n'),
            'DRUG11Q3.txt': ('$$$$$$7652730', '\n'),
            'DRUG11Q4.txt': ('021487$7941354', '\n')
        }

        for filename, (old, new) in fixes.items():
            file_path = extract_dir / 'ascii' / filename
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                content = content.replace(old, old + new)
                with open(file_path, 'w') as f:
                    f.write(content)

    def get_file_list(self) -> List[Path]:
        """Get list of all downloaded FAERS files."""
        files = []
        for txt_file in self.raw_dir.rglob('*.txt'):
            if not any(x in str(txt_file).upper() for x in ['STAT', 'SIZE']):
                files.append(txt_file)
        return files
