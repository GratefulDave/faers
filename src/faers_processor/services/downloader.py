"""Service for downloading FAERS data."""
import logging
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
import concurrent.futures
import io

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

    FAERS_URL = "https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html"
    BASE_URL = "https://fis.fda.gov/content/Exports/faers_ascii_"

    def __init__(self, output_dir: Path):
        """Initialize the FAERS downloader.
        
        Args:
            output_dir: Base directory for downloaded files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = None

    def get_quarters(self) -> List[str]:
        """Get list of available FAERS quarters.
        
        Returns:
            List of quarter identifiers (e.g., ['2023Q1', '2023Q2'])
        """
        try:
            response = requests.get(self.FAERS_URL)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=re.compile(r'.*\.zip$'))
            
            quarters = set()
            for link in links:
                match = re.search(r'(\d{4}q[1-4])', link['href'].lower())
                if match:
                    quarters.add(match.group(1))
            
            return sorted(list(quarters))
            
        except Exception as e:
            logging.error(f"Error getting quarters: {str(e)}")
            return []

    def download_quarter(self, quarter: str, output_dir: Path) -> bool:
        """Download a specific FAERS quarter.
        
        Args:
            quarter: Quarter identifier (e.g., '2023Q1')
            output_dir: Directory to save downloaded files
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Get download links for quarter
            response = requests.get(self.FAERS_URL)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=re.compile(f'.*{quarter.lower()}.*\.zip$'))
            
            if not links:
                logging.warning(f"No download links found for quarter {quarter}")
                return False
            
            # Create quarter directory
            quarter_dir = output_dir / quarter
            quarter_dir.mkdir(parents=True, exist_ok=True)
            
            # Download and extract each file
            for link in tqdm(links, desc=f"Downloading {quarter} files"):
                url = link['href']
                filename = url.split('/')[-1]
                zip_path = quarter_dir / filename
                
                # Download file
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                    with open(zip_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                # Extract file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(quarter_dir)
                
                # Remove zip file after extraction
                zip_path.unlink()
            
            return True
            
        except Exception as e:
            logging.error(f"Error downloading quarter {quarter}: {str(e)}")
            return False

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_download_links(self) -> Set[str]:
        """Get all FAERS download links."""
        async with self.session.get(self.FAERS_URL) as response:
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
                    quarter=quarter,
                    output_dir=self.output_dir
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

    def download_quarter(self, quarter: str) -> None:
        """Download and extract a specific FAERS quarter.
        
        Args:
            quarter: Quarter identifier (e.g., '23Q1')
        """
        url = f"{self.BASE_URL}{quarter}.zip"
        quarter_dir = self.output_dir / quarter
        quarter_dir.mkdir(exist_ok=True)
        
        try:
            # Download the zip file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Extract the zip file
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(quarter_dir)
            
            logging.info(f"Successfully downloaded and extracted {quarter}")
            
        except requests.RequestException as e:
            logging.error(f"Error downloading {quarter}: {str(e)}")
            raise
        except zipfile.BadZipFile as e:
            logging.error(f"Error extracting {quarter}: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error processing {quarter}: {str(e)}")
            raise


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
