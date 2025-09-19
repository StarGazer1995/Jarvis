"""
Intent Recognition and Entity Extraction Engine for ARK

This module provides intent recognition and entity extraction capabilities
for the ARK-powered Jarvis system using pattern matching and rule-based approaches.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from enum import Enum


class IntentType(Enum):
    """Enumeration of supported intent types."""
    GREETING = "greeting"
    QUESTION = "question"
    COMMAND = "command"
    REQUEST = "request"
    INFORMATION = "information"
    TOOL_USE = "tool_use"
    CONVERSATION = "conversation"
    HELP = "help"
    GOODBYE = "goodbye"
    WEATHER = "weather"
    TIME = "time"
    CALCULATION = "calculation"
    SEARCH = "search"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """Represents an extracted entity from user input."""
    name: str
    value: str
    entity_type: str
    confidence: float
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def type(self) -> str:
        """Alias for entity_type attribute for backward compatibility."""
        return self.entity_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create entity from dictionary representation."""
        return cls(
            name=data["name"],
            value=data["value"],
            entity_type=data["entity_type"],
            confidence=data["confidence"],
            start_pos=data.get("start_pos", None),
            end_pos=data.get("end_pos", None),
            metadata=data.get("metadata", {})
        )


@dataclass
class IntentResult:
    """Result of intent recognition and entity extraction."""
    intent: IntentType
    confidence: float
    entities: List[Entity]
    raw_text: str
    processed_text: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert intent result to dictionary representation."""
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "entities": [entity.to_dict() for entity in self.entities],
            "raw_text": self.raw_text,
            "processed_text": self.processed_text,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntentResult':
        """Create intent result from dictionary representation."""
        return cls(
            intent=IntentType(data["intent"]),
            confidence=data["confidence"],
            entities=[Entity.from_dict(entity_data) for entity_data in data["entities"]],
            raw_text=data["raw_text"],
            processed_text=data["processed_text"],
            metadata=data.get("metadata", {})
        )


class ARKIntentEngine:
    """
    Intent recognition and entity extraction engine for ARK.
    
    This engine uses pattern matching and rule-based approaches to identify
    user intents and extract relevant entities from natural language input.
    """
    
    def __init__(self):
        """Initialize the ARK intent engine."""
        self.ark_logger = logging.getLogger('ark.intent')
        
        # Intent patterns - mapping patterns to intent types
        self.intent_patterns = self._initialize_intent_patterns()
        
        # Entity patterns - for extracting structured information
        self.entity_patterns = self._initialize_entity_patterns()
        
        # Tool-related keywords for tool use intent detection
        self.tool_keywords = {
            'weather', 'time', 'calendar', 'email', 'search', 'calculate',
            'translate', 'convert', 'find', 'lookup', 'check', 'get', 'show',
            'tell', 'what', 'when', 'where', 'how', 'why'
        }
        
        self.ark_logger.info("ARK intent engine initialized with pattern-based recognition")
    
    def _initialize_intent_patterns(self) -> Dict[IntentType, List[str]]:
        """
        Initialize intent recognition patterns.
        
        Returns:
            Dictionary mapping intent types to regex patterns
        """
        return {
            IntentType.GREETING: [
                r'\b(hello|hi|hey|good morning|good afternoon|good evening|greetings)\b',
                r'\b(howdy|what\'s up|how do you do)\b',
                r'\b(hi|hello|hey).*how are you.*\?',
                r'\bhow are you\b'
            ],
            IntentType.GOODBYE: [
                r'\b(goodbye|bye|see you|farewell|take care|until next time)\b',
                r'\b(good night|have a good day|catch you later)\b',
                r'\b(talk to you|speak to you).*(soon|later)\b'
            ],
            IntentType.QUESTION: [
                r'^\s*(what|when|where|who|why|how|which|can you|could you|would you)\b',
                r'\b(tell me|explain|describe|what is|what are)\b',
                r'.*\?$'  # Ends with question mark
            ],
            IntentType.COMMAND: [
                r'^\s*(do|run|execute|start|stop|create|delete|update|modify)\b',
                r'^\s*(please\s+)?(make|build|generate|produce|write)\b',
                r'^\s*(open|close|save|load|import|export)\b',
                r'\b(show|display|view).*\b(calendar|schedule|agenda|appointments)\b',
                r'\b(help me|show me)\b'
            ],
            IntentType.REQUEST: [
                r'\b(please|could you|would you|can you|i need|i want|i would like)\b',
                r'\b(assist me|give me)\b'
            ],
            IntentType.TOOL_USE: [
                r'\b(weather|temperature|forecast)\b',
                r'\b(time|date|calendar|schedule)\b',
                r'\b(search|lookup|google)\b',
                r'\b(calculate|compute|math|convert)\b',
                r'\b(translate|translation)\b'
            ],
            IntentType.HELP: [
                r'\b(what can you do|capabilities)\b',
                r'^\s*(help|\\?|\\help)\s*$',
                r'\b(need|want|require).*\b(help|assistance|support)\b',
                r'\b(can you help|need help|want help)\b'
            ],
            IntentType.INFORMATION: [
                r'\b(information|info|details|facts|data)\b',
                r'\b(status|report|summary|overview)\b'
            ],
            IntentType.WEATHER: [
                r'\b(weather|temperature|forecast|climate)\b',
                r'\b(how\'s the weather|what\'s the weather|weather like)\b',
                r'\b(rain|snow|sunny|cloudy|storm|wind)\b',
                r'\b(degrees|celsius|fahrenheit|humidity)\b',
                r'\b(is it|will it be).*(raining|snowing|sunny|cloudy|windy|hot|cold)\b'
            ],
            IntentType.TIME: [
                r'\b(time|date|clock|hour|minute|second)\b',
                r'\b(what time|current time|what\'s the time)\b',
                r'\b(what.*time.*now|what.*time.*is.*it|what.*date.*today|what.*day.*today)\b',
                r'\b(show.*time|tell.*time|time.*is)\b',
                r'\b(what.*schedule|when.*appointment)\b'
            ],
            IntentType.CALCULATION: [
                r'\b(calculate|compute|math|add|subtract|multiply|divide)\b',
                r'\b(what is|what\'s)\s+\d+\s*[+\-*/]\s*\d+',
                r'\d+\s*[+\-*/]\s*\d+',
                r'\b(sum|total|average|mean)\b'
            ],
            IntentType.SEARCH: [
                r'\b(search|lookup|google|bing|look up)\b',
                r'\b(search for|find me|look for)\b',
                r'^(find|search|google|lookup)\s+\w+'
            ]
        }
    
    def _initialize_entity_patterns(self) -> Dict[str, Union[str, List[str]]]:
        """
        Initialize entity extraction patterns.
        
        Returns:
            Dictionary mapping entity types to regex patterns
        """
        return {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?',
            'date': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
            'time': r'\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?:\s*(?:AM|PM|am|pm))?\b',
            'number': r'\b\d+(?:\.\d+)?\b',
            'currency': r'\$\d+(?:\.\d{2})?|\b\d+(?:\.\d{2})?\s*(?:dollars?|USD|euros?|EUR)\b',
            'LOCATION': [
                r'\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose|Austin|Jacksonville|Fort Worth|Columbus|Charlotte|San Francisco|Indianapolis|Seattle|Denver|Washington|Boston|El Paso|Nashville|Detroit|Oklahoma City|Portland|Las Vegas|Memphis|Louisville|Baltimore|Milwaukee|Albuquerque|Tucson|Fresno|Sacramento|Kansas City|Long Beach|Mesa|Atlanta|Colorado Springs|Virginia Beach|Raleigh|Omaha|Miami|Oakland|Minneapolis|Tulsa|Wichita|New Orleans|Arlington|Cleveland|Bakersfield|Tampa|Aurora|Honolulu|Anaheim|Santa Ana|Corpus Christi|Riverside|Lexington|Stockton|Toledo|St. Paul|Newark|Greensboro|Plano|Henderson|Lincoln|Buffalo|Jersey City|Chula Vista|Fort Wayne|Orlando|St. Petersburg|Chandler|Laredo|Norfolk|Durham|Madison|Lubbock|Irvine|Winston-Salem|Glendale|Garland|Hialeah|Reno|Chesapeake|Gilbert|Baton Rouge|Irving|Scottsdale|North Las Vegas|Fremont|Boise|Richmond|San Bernardino|Birmingham|Spokane|Rochester|Des Moines|Modesto|Fayetteville|Tacoma|Oxnard|Fontana|Columbus|Montgomery|Moreno Valley|Shreveport|Aurora|Yonkers|Akron|Huntington Beach|Little Rock|Augusta|Amarillo|Glendale|Mobile|Grand Rapids|Salt Lake City|Tallahassee|Huntsville|Grand Prairie|Knoxville|Worcester|Newport News|Brownsville|Overland Park|Santa Clarita|Providence|Garden Grove|Chattanooga|Oceanside|Jackson|Fort Lauderdale|Santa Rosa|Rancho Cucamonga|Port St. Lucie|Tempe|Ontario|Vancouver|Cape Coral|Sioux Falls|Springfield|Peoria|Pembroke Pines|Elk Grove|Salem|Lancaster|Corona|Eugene|Palmdale|Salinas|Springfield|Pasadena|Fort Collins|Hayward|Pomona|Cary|Rockford|Alexandria|Escondido|McKinney|Kansas City|Joliet|Sunnyvale|Torrance|Bridgeport|Lakewood|Hollywood|Paterson|Naperville|Syracuse|Mesquite|Dayton|Savannah|Clarksville|Orange|Pasadena|Fullerton|Killeen|Frisco|Hampton|McAllen|Warren|Bellevue|West Valley City|Columbia|Olathe|Sterling Heights|New Haven|Miramar|Waco|Thousand Oaks|Cedar Rapids|Charleston|Sioux City|Round Rock|Fargo|Carrollton|Roseville|Concord|Thornton|Visalia|Gainesville|Coral Springs|Stamford|Westminster|Elizabeth|Macon|Abilene|Beaumont|Independence|Murfreesboro|Ann Arbor|Springfield|Berkeley|Peoria|Provo|El Monte|Columbia|Lansing|Fargo|Downey|Costa Mesa|Wilmington|Arvada|Inglewood|Miami Gardens|Carlsbad|Westminster|Rochester|Odessa|Manchester|Elgin|West Jordan|Round Rock|Clearwater|Waterbury|Gresham|Fairfield|Billings|Lowell|San Buenaventura|Pueblo|High Point|West Covina|Richmond|Murrieta|Cambridge|Antioch|Temecula|Norwalk|Centennial|Everett|Palm Bay|Wichita Falls|Green Bay|Daly City|Burbank|Richardson|Pompano Beach|North Charleston|Broken Arrow|Boulder|West Palm Beach|Surprise|Thornton|League City|Lakeland|Edison|Tyler|Pearland|College Station|Kenosha|Allentown|Hillsboro|Rochester|Compton|Tuscaloosa|St. Joseph|Bellingham|Spokane Valley|Paris|London|Tokyo|Berlin|Madrid|Rome|Moscow|Beijing|Sydney|Toronto|Montreal|Vancouver|Mexico City|Cairo|Mumbai|Delhi|Bangkok|Singapore|Dubai|Istanbul|Amsterdam|Stockholm|Copenhagen|Oslo|Helsinki|Vienna|Prague|Budapest|Warsaw|Zurich|Geneva|Brussels|Lisbon|Athens|Dublin|Edinburgh|Cardiff|Belfast)\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b(?=\s+(?:city|town|village|county|state|country|province|region))',
                r'(?:in|at|near|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
                r'\b[A-Z][a-z]+\b(?=\s+and\s+[A-Z][a-z]+)',  # Capture locations before "and"
                r'(?:and\s+)([A-Z][a-z]+)\b'  # Capture locations after "and"
            ],
            'TIME': [
                r'\b(today|tomorrow|yesterday|tonight|morning|afternoon|evening|night|now|later|soon|early|late)\b',
                r'\b(?:at\s+)?(?:\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?|\d{1,2}\s*[AaPp][Mm])\b',
                r'\b(?:in\s+)?(?:\d+\s+(?:minutes?|hours?|days?|weeks?|months?|years?))\b',
                r'\b(?:next|last)\s+(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
                r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b',
                r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b',
                r'\b\d{4}-\d{2}-\d{2}\b'
            ],
            'LOCATION': [
                r'\b(New York|Paris|London|Berlin|Madrid|Rome|Amsterdam|Brussels|Vienna|Prague|Budapest|Warsaw|Stockholm|Oslo|Copenhagen|Helsinki|Dublin|Edinburgh|Glasgow|Manchester|Liverpool|Birmingham|Leeds|Sheffield|Bristol|Cardiff|Belfast|Tokyo|Osaka|Kyoto|Beijing|Shanghai|Mumbai|Delhi|Moscow|Saint Petersburg)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b(?=\s+and\s+[A-Z][a-z]+\s+(?:today|tomorrow|yesterday|tonight|morning|afternoon|evening|night))',
                r'(?<=and\s)([A-Z][a-z]+)\b(?=\s+(?:today|tomorrow|yesterday|tonight|morning|afternoon|evening|night))'
            ],
            'PERSON': r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'FILE_PATH': r'(?:[a-zA-Z]:)?[/\\]?(?:[^/\\:\*\?"<>\|]+[/\\])*[^/\\:\*\?"<>\|]*\.[a-zA-Z0-9]+',
            'COMMAND': r'\b(?:run|execute|start|stop|kill|restart)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
        }
    
    def recognize_intent(self, text: str) -> IntentResult:
        """
        Recognize intent and extract entities from user input.
        
        Args:
            text: User input text
            
        Returns:
            IntentResult with recognized intent and extracted entities
        """
        import time
        start_time = time.time()
        
        if not text or not text.strip():
            return IntentResult(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                entities=[],
                raw_text=text,
                processed_text=""
            )
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Recognize intent
        intent, confidence = self._match_intent(processed_text)
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create result
        result = IntentResult(
            intent=intent,
            confidence=confidence,
            entities=entities,
            raw_text=text,
            processed_text=processed_text,
            metadata={
                "processing_timestamp": self._get_timestamp(),
                "matched_patterns": self._get_pattern_matches(processed_text),
                "processing_time": processing_time,
                "entity_count": len(entities)
            }
        )
        
        self.ark_logger.debug(f"ARK intent: Recognized '{intent.value}' (confidence: {confidence:.2f}) with {len(entities)} entities")
        
        return result
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Public method to extract entities from text.
        
        Args:
            text: Raw input text
            
        Returns:
            List of extracted entities
        """
        return self._extract_entities(text)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for intent recognition.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        processed = text.lower().strip()
        
        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        # Remove common punctuation that doesn't affect intent
        processed = re.sub(r'[,;]', ' ', processed)
        
        return processed
    
    def _match_intent(self, text: str) -> Tuple[IntentType, float]:
        """
        Match text against intent patterns.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Tuple of (intent_type, confidence_score)
        """
        best_intent = IntentType.UNKNOWN
        best_confidence = 0.0
        
        # Define intent priority (higher priority intents are checked first)
        intent_priority = {
            IntentType.WEATHER: 10,
            IntentType.TIME: 10,
            IntentType.CALCULATION: 9,
            IntentType.SEARCH: 9,
            IntentType.GREETING: 9,
            IntentType.GOODBYE: 9,
            IntentType.HELP: 8,  # Help requests should have higher priority than general requests
            IntentType.COMMAND: 8,
            IntentType.REQUEST: 7,
            IntentType.TOOL_USE: 6,
            IntentType.INFORMATION: 5,
            IntentType.CONVERSATION: 4,
            IntentType.QUESTION: 3,  # Lower priority for general questions
            IntentType.UNKNOWN: 1
        }
        
        # Sort intents by priority
        sorted_intents = sorted(self.intent_patterns.items(), 
                              key=lambda x: intent_priority.get(x[0], 1), 
                              reverse=True)
        
        # Collect all matches with their priorities and confidences
        all_matches = []
        
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Calculate confidence based on match quality
                    match_length = len(match.group(0))
                    text_length = len(text)
                    
                    # Base confidence calculation
                    if match_length == text_length:
                        # Exact match gets high confidence
                        base_confidence = 0.9
                    elif match_length >= text_length * 0.8:
                        # Strong partial match
                        base_confidence = 0.8
                    elif match_length >= text_length * 0.5:
                        # Good partial match
                        base_confidence = 0.6
                    else:
                        # For shorter matches, check if it's a complete word boundary match
                        matched_text = match.group(0).strip()
                        # If the matched text is a complete meaningful word (>= 3 chars)
                        # and represents a significant portion, give higher confidence
                        if len(matched_text) >= 3 and match_length >= 4:
                            # Check if the match is at the beginning of the text (stronger signal)
                            if match.start() <= 2:  # Allow for minor leading whitespace
                                base_confidence = 0.7
                            else:
                                # Match in the middle or end is weaker
                                base_confidence = 0.5
                        else:
                            # Weak match
                            base_confidence = min(0.5, match_length / text_length + 0.2)
                    
                    priority = intent_priority.get(intent_type, 1)
                    all_matches.append((intent_type, base_confidence, priority, match_length))
        
        # Select the best match: first by priority, then by confidence
        if all_matches:
            # Sort by priority (descending), then by confidence (descending)
            all_matches.sort(key=lambda x: (x[2], x[1]), reverse=True)
            best_intent = all_matches[0][0]
            base_confidence = all_matches[0][1]
            priority = all_matches[0][2]
            
            # Add priority bonus for final confidence
            priority_bonus = priority * 0.02
            best_confidence = min(0.95, base_confidence + priority_bonus)
        
        # Special handling for tool use detection
        if best_intent == IntentType.UNKNOWN:
            tool_confidence = self._detect_tool_intent(text)
            if tool_confidence > 0.5:
                best_intent = IntentType.TOOL_USE
                best_confidence = tool_confidence
        
        # Default to unknown if no strong match
        if best_confidence < 0.3:
            best_intent = IntentType.UNKNOWN
            best_confidence = 0.1
        
        return best_intent, best_confidence
    
    def _detect_tool_intent(self, text: str) -> float:
        """
        Detect if the text indicates tool usage intent.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Confidence score for tool use intent
        """
        words = set(text.split())
        tool_word_matches = words.intersection(self.tool_keywords)
        
        if not tool_word_matches:
            return 0.0
        
        # Calculate confidence based on tool keyword density
        confidence = len(tool_word_matches) / len(words)
        return min(0.8, confidence * 2)  # Cap at 0.8, boost by 2x
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text using pattern matching.
        
        Args:
            text: Raw input text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            # Handle both string patterns and lists of patterns
            patterns = pattern if isinstance(pattern, list) else [pattern]
            
            for p in patterns:
                matches = re.finditer(p, text, re.IGNORECASE)
                
                for match in matches:
                    entity = Entity(
                        name=entity_type.lower(),
                        value=match.group(0),
                        entity_type=entity_type,
                        confidence=0.9,  # High confidence for regex matches
                        start_pos=match.start(),
                        end_pos=match.end(),
                        metadata={
                            "pattern": p,
                            "match_groups": match.groups()
                        }
                    )
                    entities.append(entity)
        
        # Remove overlapping entities (keep the longer ones)
        entities = self._remove_overlapping_entities(entities)
        
        return entities
    
    def _remove_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Remove overlapping entities, keeping the longer ones.
        
        Args:
            entities: List of entities that may overlap
            
        Returns:
            List of non-overlapping entities
        """
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda e: e.start_pos)
        
        filtered = []
        for entity in entities:
            # Check if this entity overlaps with any already filtered entity
            overlaps = False
            for existing in filtered:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    # There's an overlap - keep the longer entity
                    if len(entity.value) > len(existing.value):
                        filtered.remove(existing)
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Public method to extract entities from text.
        
        Args:
            text: Raw input text
            
        Returns:
            List of extracted entities
        """
        return self._extract_entities(text)
    
    def _get_pattern_matches(self, text: str) -> Dict[str, List[str]]:
        """
        Get all pattern matches for debugging/metadata.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Dictionary of intent types to matched patterns
        """
        matches = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            intent_matches = []
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    intent_matches.append(pattern)
            
            if intent_matches:
                matches[intent_type.value] = intent_matches
        
        return matches
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def add_custom_intent_pattern(self, intent_type: IntentType, pattern: str) -> None:
        """
        Add a custom intent pattern.
        
        Args:
            intent_type: The intent type to add the pattern for
            pattern: Regex pattern to match
        """
        if intent_type not in self.intent_patterns:
            self.intent_patterns[intent_type] = []
        
        self.intent_patterns[intent_type].append(pattern)
        self.ark_logger.info(f"ARK intent: Added custom pattern for {intent_type.value}: {pattern}")
    
    def add_custom_entity_pattern(self, entity_type: str, pattern: str) -> None:
        """
        Add a custom entity extraction pattern.
        
        Args:
            entity_type: The entity type name
            pattern: Regex pattern to extract the entity
        """
        self.entity_patterns[entity_type] = pattern
        self.ark_logger.info(f"ARK intent: Added custom entity pattern for {entity_type}: {pattern}")
    
    def get_supported_intents(self) -> List[str]:
        """
        Get list of supported intent types.
        
        Returns:
            List of supported intent type names
        """
        return [intent.value for intent in IntentType]
    
    def get_supported_entities(self) -> List[str]:
        """
        Get list of supported entity types.
        
        Returns:
            List of supported entity type names
        """
        return list(self.entity_patterns.keys())
    
    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """
        Analyze the complexity of input text for processing insights.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with complexity metrics
        """
        words = text.split()
        sentences = text.split('.')
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "question_marks": text.count('?'),
            "exclamation_marks": text.count('!'),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            "special_chars": len(re.findall(r'[^a-zA-Z0-9\s]', text)),
            "complexity_score": self._calculate_complexity_score(text)
        }
    
    def _calculate_complexity_score(self, text: str) -> float:
        """
        Calculate a complexity score for the text.
        
        Args:
            text: Input text
            
        Returns:
            Complexity score between 0 and 1
        """
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # Factors that increase complexity
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentence_count = len(text.split('.'))
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / len(text)
        
        # Normalize and combine factors
        complexity = (
            min(avg_word_length / 10, 0.4) +  # Word length factor (max 0.4)
            min(len(words) / 50, 0.3) +       # Length factor (max 0.3)
            min(special_char_ratio * 2, 0.3)  # Special char factor (max 0.3)
        )
        
        return min(complexity, 1.0)