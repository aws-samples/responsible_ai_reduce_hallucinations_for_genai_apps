# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Pydantic type models for Bedrock Guardrails APIs

These classes help us parse and provide IDE autocomplete for Bedrock Guardrails API results.

See the Bedrock API reference:
- https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
- https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Types_Amazon_Bedrock_Runtime.html
"""

# Python Built-Ins:
from enum import Enum
from typing import Literal

# External Dependencies:
from pydantic import BaseModel, Field


class ApiGuardrailAction(str, Enum):
    NONE = "NONE"
    INTERVENED = "GUARDRAIL_INTERVENED"


class ApiGuardrailConfidence(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ApiGuardrailAutomatedReasoningRule(BaseModel):
    # Both optional technically
    identifier: str
    policy_version_arn: str = Field(alias="policyVersionArn")


class ApiGuardrailAutomatedReasoningLogicWarningType(str, Enum):
    ALWAYS_FALSE = "ALWAYS_FALSE"
    ALWAYS_TRUE = "ALWAYS_TRUE"


class ApiGuardrailAutomatedReasoningStatement(BaseModel):
    # Both optional technically
    logic: str
    natural_language: str = Field(alias="naturalLanguage")


class ApiGuardrailAutomatedReasoningLogicWarning(BaseModel):
    claims: list[ApiGuardrailAutomatedReasoningStatement]
    premises: list[ApiGuardrailAutomatedReasoningStatement]
    type: ApiGuardrailAutomatedReasoningLogicWarningType


class ApiGuardrailAutomatedReasoningInputTextReference(BaseModel):
    text: str


class ApiGuardrailAutomatedReasoningTranslation(BaseModel):
    # All optional technically
    claims: list[ApiGuardrailAutomatedReasoningStatement]
    confidence: float
    premises: list[ApiGuardrailAutomatedReasoningStatement]
    untranslated_claims: list[ApiGuardrailAutomatedReasoningInputTextReference] = Field(
        alias="untranslatedClaims"
    )
    untranslated_premises: list[ApiGuardrailAutomatedReasoningInputTextReference] = (
        Field(alias="untranslatedPremises")
    )


class ApiGuardrailAutomatedReasoningImpossibleOrInvalidFinding(BaseModel):
    contradicting_rules: list[ApiGuardrailAutomatedReasoningRule] | None = Field(
        alias="contradictingRules", default=None
    )
    logic_warning: ApiGuardrailAutomatedReasoningLogicWarning | None = Field(
        alias="logicWarning"
    )
    translation: ApiGuardrailAutomatedReasoningTranslation | None = None


class ApiGuardrailAutomatedReasoningFindingWithNoData(BaseModel):
    pass


class ApiGuardrailAutomatedReasoningScenario(BaseModel):
    statements: list[ApiGuardrailAutomatedReasoningStatement]


class ApiGuardrailAutomatedReasoningSatisfiableFinding(BaseModel):
    claims_false_scenario: ApiGuardrailAutomatedReasoningScenario | None = Field(
        alias="claimsFalseScenario", default=None
    )
    claims_true_scenario: ApiGuardrailAutomatedReasoningScenario | None = Field(
        alias="claimsTrueScenario", default=None
    )
    logic_warning: ApiGuardrailAutomatedReasoningLogicWarning | None = Field(
        alias="logicWarning", default=None
    )
    translation: ApiGuardrailAutomatedReasoningTranslation | None = None


class ApiGuardrailAutomatedReasoningTranslationOption(BaseModel):
    translations: list[ApiGuardrailAutomatedReasoningTranslation]


class ApiGuardrailAutomatedReasoningTranslationAmbiguousFinding(BaseModel):
    difference_scenarios: list[ApiGuardrailAutomatedReasoningScenario] | None = Field(
        alias="differenceScenarios", default=None
    )
    options: list[ApiGuardrailAutomatedReasoningTranslationOption] | None = None


class ApiGuardrailAutomatedReasoningValidFinding(BaseModel):
    claims_true_scenario: ApiGuardrailAutomatedReasoningScenario | None = Field(
        alias="claimsTrueScenario", default=None
    )
    logic_warning: ApiGuardrailAutomatedReasoningLogicWarning | None = Field(
        alias="logicWarning", default=None
    )
    supporting_rules: list[ApiGuardrailAutomatedReasoningRule] | None = Field(
        alias="supportingRules", default=None
    )
    translation: ApiGuardrailAutomatedReasoningTranslation | None = None


class ApiGuardrailAutomatedReasoningFinding(BaseModel):
    # TODO: This data type is a UNION
    impossible: ApiGuardrailAutomatedReasoningImpossibleOrInvalidFinding | None = None
    invalid: ApiGuardrailAutomatedReasoningImpossibleOrInvalidFinding | None = None
    no_translations: ApiGuardrailAutomatedReasoningFindingWithNoData | None = Field(
        alias="noTranslations", default=None
    )
    satisfiable: ApiGuardrailAutomatedReasoningSatisfiableFinding | None = None
    too_complex: ApiGuardrailAutomatedReasoningFindingWithNoData | None = Field(
        alias="tooComplex", default=None
    )
    translation_ambiguous: (
        ApiGuardrailAutomatedReasoningTranslationAmbiguousFinding | None
    ) = Field(alias="translationAmbiguous", default=None)
    valid: ApiGuardrailAutomatedReasoningValidFinding | None = None


class ApiGuardrailAutomatedReasoningPolicyAssessment(BaseModel):
    findings: list[ApiGuardrailAutomatedReasoningFinding]


class ApiGuardrailContentFilterType(str, Enum):
    INSULTS = "INSULTS"
    HATE = "HATE"
    SEXUAL = "SEXUAL"
    VIOLENCE = "VIOLENCE"
    MISCONDUCT = "MISCONDUCT"
    PROMPT_ATTACK = "PROMPT_ATTACK"


class ApiGuardrailFilterAction(str, Enum):
    BLOCKED = "BLOCKED"
    NONE = "NONE"


class ApiGuardrailContentFilter(BaseModel):
    action: ApiGuardrailFilterAction
    confidence: ApiGuardrailConfidence
    type: ApiGuardrailContentFilterType
    detected: bool | None = None
    filter_strength: ApiGuardrailConfidence


class ApiGuardrailContentPolicyAssessment(BaseModel):
    filters: list[ApiGuardrailContentFilter]


class ApiGuardrailContextualGroundingFilterType(str, Enum):
    GROUNDING = "GROUNDING"
    RELEVANCE = "RELEVANCE"


class ApiGuardrailContextualGroundingFilter(BaseModel):
    action: ApiGuardrailFilterAction
    detected: bool | None = None
    score: float
    threshold: float
    type: ApiGuardrailContextualGroundingFilterType


class ApiGuardrailContextualGroundingPolicyAssessment(BaseModel):
    filters: list[ApiGuardrailContextualGroundingFilter]


class ApiGuardrailItemCoverage(BaseModel):
    guarded: int
    total: int


class ApiGuardrailCoverage(BaseModel):
    images: ApiGuardrailItemCoverage | None = None
    text_characters: ApiGuardrailItemCoverage | None = Field(
        alias="textCharacters", default=None
    )


class ApiGuardrailUsage(BaseModel):
    automated_reasoning_policies: int = Field(alias="automatedReasoningPolicies")
    content_policy_image_units: int = Field(alias="contentPolicyImageUnits")
    content_policy_units: int = Field(alias="contentPolicyUnits")
    contextual_grounding_policy_units: int = Field(
        alias="contextualGroundingPolicyUnits"
    )
    sensitive_information_policy_free_units: int = Field(
        alias="sensitiveInformationPolicyFreeUnits"
    )
    sensitive_information_policy_units: int = Field(
        alias="sensitiveInformationPolicyUnits"
    )
    topic_policy_units: int = Field(alias="topicPolicyUnits")
    word_policy_units: int = Field(alias="wordPolicyUnits")


class ApiGuardrailInvocationMetrics(BaseModel):
    coverage: ApiGuardrailCoverage = Field(alias="guardrailCoverage")
    latency: int | None = None
    usage: ApiGuardrailUsage


class ApiPiiFilterAction(str, Enum):
    ANONYMIZED = "ANONYMIZED"
    BLOCKED = "BLOCKED"
    NONE = "NONE"


class ApiPiiEntityType(str, Enum):
    ADDRESS = "ADDRESS"
    AGE = "AGE"
    AWS_ACCESS_KEY = "AWS_ACCESS_KEY"
    AWS_SECRET_KEY = "AWS_SECRET_KEY"
    CA_HEALTH_NUMBER = "CA_HEALTH_NUMBER"
    CA_SOCIAL_INSURANCE_NUMBER = "CA_SOCIAL_INSURANCE_NUMBER"
    CREDIT_DEBIT_CARD_CVV = "CREDIT_DEBIT_CARD_CVV"
    CREDIT_DEBIT_CARD_EXPIRY = "CREDIT_DEBIT_CARD_EXPIRY"
    CREDIT_DEBIT_CARD_NUMBER = "CREDIT_DEBIT_CARD_NUMBER"
    DRIVER_ID = "DRIVER_ID"
    EMAIL = "EMAIL"
    INTERNATIONAL_BANK_ACCOUNT_NUMBER = "INTERNATIONAL_BANK_ACCOUNT_NUMBER"
    IP_ADDRESS = "IP_ADDRESS"
    LICENSE_PLATE = "LICENSE_PLATE"
    MAC_ADDRESS = "MAC_ADDRESS"
    NAME = "NAME"
    PASSWORD = "PASSWORD"
    PHONE = "PHONE"
    PIN = "PIN"
    SWIFT_CODE = "SWIFT_CODE"
    UK_NATIONAL_HEALTH_SERVICE_NUMBER = "UK_NATIONAL_HEALTH_SERVICE_NUMBER"
    UK_NATIONAL_INSURANCE_NUMBER = "UK_NATIONAL_INSURANCE_NUMBER"
    UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER = "UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER"
    URL = "URL"
    USERNAME = "USERNAME"
    US_BANK_ACCOUNT_NUMBER = "US_BANK_ACCOUNT_NUMBER"
    US_BANK_ROUTING_NUMBER = "US_BANK_ROUTING_NUMBER"
    US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER = "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER"
    US_PASSPORT_NUMBER = "US_PASSPORT_NUMBER"
    US_SOCIAL_SECURITY_NUMBER = "US_SOCIAL_SECURITY_NUMBER"
    VEHICLE_IDENTIFICATION_NUMBER = "VEHICLE_IDENTIFICATION_NUMBER"


class ApiGuardrailPiiEntityFilter(BaseModel):
    action: ApiPiiFilterAction
    detected: bool | None = None
    match: str
    type: ApiPiiEntityType


class ApiGuardrailRegexFilter(BaseModel):
    action: ApiPiiFilterAction
    detected: bool | None = None
    match: str | None = None
    name: str | None = None
    regex: str | None = None


class ApiGuardrailSensitiveInformationPolicyAssessment(BaseModel):
    pii_entities: list[ApiGuardrailPiiEntityFilter] = Field(alias="piiEntities")
    regexes: list[ApiGuardrailRegexFilter]


class ApiGuardrailTopic(BaseModel):
    action: ApiGuardrailFilterAction
    name: str
    type: Literal["DENY"]
    detected: bool | None = None


class ApiGuardrailTopicPolicyAssessment(BaseModel):
    topics: list[ApiGuardrailTopic]


class ApiGuardrailCustomWord(BaseModel):
    action: ApiGuardrailFilterAction
    match: str
    detected: bool | None = None


class ApiGuardrailManagedWord(BaseModel):
    action: ApiGuardrailFilterAction
    match: str
    type: Literal["PROFANITY"]
    detected: bool | None = None


class ApiGuardrailWordPolicyAssessment(BaseModel):
    custom_words: list[ApiGuardrailCustomWord] = Field(alias="customWords")
    managed_word_lists: list[ApiGuardrailManagedWord] = Field(alias="managedWordLists")


class ApiGuardrailAssessment(BaseModel):
    automated_reasoning_policy: (
        ApiGuardrailAutomatedReasoningPolicyAssessment | None
    ) = Field(alias="automatedReasoningPolicy", default=None)
    content_policy: ApiGuardrailContentPolicyAssessment | None = Field(
        alias="contentPolicy", default=None
    )
    contextual_grounding_policy: (
        ApiGuardrailContextualGroundingPolicyAssessment | None
    ) = Field(alias="contextualGroundingPolicy", default=None)
    invocation_metrics: ApiGuardrailInvocationMetrics | None = Field(
        alias="invocationMetrics", default=None
    )
    sensitive_information_policy: (
        ApiGuardrailSensitiveInformationPolicyAssessment | None
    ) = Field(alias="sensitiveInformationPolicy", default=None)
    topic_policy: ApiGuardrailTopicPolicyAssessment | None = Field(
        alias="topicPolicy", default=None
    )
    word_policy: ApiGuardrailWordPolicyAssessment | None = Field(
        alias="wordPolicy", default=None
    )


class ApiGuardrailOutput(BaseModel):
    text: str


class ApiApplyGuardrailResponse(BaseModel):
    action: ApiGuardrailAction
    action_reason: str | None = Field(alias="actionReason", default=None)
    assessments: list[ApiGuardrailAssessment]
    coverage: ApiGuardrailCoverage = Field(alias="guardrailCoverage")
    outputs: list[ApiGuardrailOutput] | None = None
    usage: ApiGuardrailUsage
