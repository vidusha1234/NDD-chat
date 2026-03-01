import re
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from models import Course, CourseChunk


@dataclass
class SearchResults:
    """Container for search results with metadata"""
    documents: List[str]
    metadata: List[Dict[str, Any]]
    distances: List[float]
    error: Optional[str] = None

    @classmethod
    def empty(cls, error_msg: str) -> 'SearchResults':
        return cls(documents=[], metadata=[], distances=[], error=error_msg)

    def is_empty(self) -> bool:
        return len(self.documents) == 0


class VectorStore:
    """BM25 keyword search store for course content and metadata"""

    def __init__(self, chroma_path: str, embedding_model: str, hf_api_key: str, max_results: int = 5):
        self.max_results = max_results
        self._chunks: List[Dict] = []       # [{text, metadata}]
        self._catalog: Dict[str, Dict] = {} # course_title -> metadata
        self._bm25: Optional[BM25Okapi] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def _rebuild_index(self):
        if self._chunks:
            corpus = [self._tokenize(c['text']) for c in self._chunks]
            self._bm25 = BM25Okapi(corpus)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, course_name: Optional[str] = None,
               lesson_number: Optional[int] = None,
               limit: Optional[int] = None) -> SearchResults:
        if not self._bm25 or not self._chunks:
            return SearchResults.empty("No documents indexed yet")

        n = limit if limit is not None else self.max_results
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        documents, metadatas, distances = [], [], []
        for idx in top_indices:
            if len(documents) >= n:
                break
            chunk = self._chunks[idx]
            meta = chunk['metadata']
            if course_name and meta.get('course_title') != course_name:
                continue
            if lesson_number is not None and meta.get('lesson_number') != lesson_number:
                continue
            documents.append(chunk['text'])
            metadatas.append(meta)
            distances.append(float(scores[idx]))

        return SearchResults(documents=documents, metadata=metadatas, distances=distances)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_course_metadata(self, course: Course):
        self._catalog[course.title] = {
            'title': course.title,
            'instructor': course.instructor,
            'course_link': course.course_link,
            'lessons': [
                {
                    'lesson_number': l.lesson_number,
                    'lesson_title': l.title,
                    'lesson_link': l.lesson_link,
                }
                for l in course.lessons
            ],
            'lesson_count': len(course.lessons),
        }

    def add_course_content(self, chunks: List[CourseChunk]):
        for chunk in chunks:
            meta: Dict[str, Any] = {
                'course_title': chunk.course_title,
                'chunk_index': chunk.chunk_index,
            }
            if chunk.lesson_number is not None:
                meta['lesson_number'] = chunk.lesson_number
            self._chunks.append({'text': chunk.content, 'metadata': meta})
        self._rebuild_index()

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def clear_all_data(self):
        self._chunks = []
        self._catalog = {}
        self._bm25 = None

    def get_existing_course_titles(self) -> List[str]:
        return list(self._catalog.keys())

    def get_course_count(self) -> int:
        return len(self._catalog)

    def get_all_courses_metadata(self) -> List[Dict[str, Any]]:
        return list(self._catalog.values())

    def get_course_link(self, course_title: str) -> Optional[str]:
        course = self._catalog.get(course_title)
        return course.get('course_link') if course else None

    def get_lesson_link(self, course_title: str, lesson_number: int) -> Optional[str]:
        course = self._catalog.get(course_title)
        if course:
            for lesson in course.get('lessons', []):
                if lesson.get('lesson_number') == lesson_number:
                    return lesson.get('lesson_link')
        return None
