# Redacted Transcript Samples for Interview Process

## Overview
This directory contains 60 anonymized files (20 samples × 3 files each) for use in interview processes. All PII has been manually redacted.

## File Structure

### Total Files: 60
- **20 Transcripts** (redacted conversation transcripts)
- **20 Summaries** (redacted AI-generated summaries)
- **20 Feedback** (redacted human feedback on summaries)

### Classification Breakdown

#### GOOD (10 samples - Good Summary)
- **good-1 to good-5**: Claims samples (5 samples)
- **good-6 to good-10**: Retail samples (5 samples)
- **Issue Type**: Good Summary
- **Use Case**: Examples of high-quality AI summaries

#### OKAY (5 samples - Minor Issues)
- **okay-1 to okay-5**: Claims samples (5 samples)
- **Issue Type**: Summarisation Issue (Minor criticality)
- **Use Case**: Examples with minor issues like spelling errors, missing non-critical details

#### BAD (5 samples - Major Issues)
- **bad-1 to bad-5**: Claims samples (5 samples)
- **Issue Type**: Summarisation Issue (Major/Critical criticality)
- **Use Case**: Examples with significant errors like wrong information, misidentified parties

## Naming Convention

Each sample has 3 files:
- `{state}-{index}-transcript.txt` - The full conversation transcript
- `{state}-{index}-summary.txt` - The AI-generated summary
- `{state}-{index}-feedback.txt` - Human reviewer feedback

Where:
- `{state}` = good, okay, or bad
- `{index}` = 1-10 for good, 1-5 for okay/bad

## Redaction Approach

All PII was manually redacted using LLM contextual understanding, including:

### Personal Information
- Names (agents, customers, callers, third parties)
- Dates of birth
- Phone numbers
- Email addresses
- Physical addresses

### Financial Information
- Claim numbers
- Policy numbers
- Reference numbers
- Financial amounts
- IBAN/Bank account numbers

### Vehicle & Location Data
- Vehicle registrations
- Specific locations (cities, streets, postcodes)
- Company names

### Placeholder Format
Placeholders follow the pattern: `[CATEGORY_NUMBER]`

Examples:
- `[AGENT_1]`, `[AGENT_2]` - Agent names
- `[CALLER_1]`, `[CUSTOMER_1]` - External parties
- `[CLAIM_NUMBER_1]` - Claim references
- `[DATE_1]`, `[TIME_1]` - Temporal data
- `[AMOUNT_1]` - Financial amounts
- `[IBAN_1]` - Banking information
- `[ADDRESS_1]`, `[LOCATION_1]` - Geographic data

### Consistency
Within each sample, the same entity always uses the same placeholder token to maintain context and readability.

## Data Sources

- **Claims data**: /OneDrive_1_11-01-2026/claims.xlsx (393 rows)
- **Retail data**: /OneDrive_1_11-01-2026/retail.xlsx (143 rows)
- **Transcripts**: /01 Alpha/Claims & Retail/01 Transcripts/

## Sample Distribution

| State | Claims | Retail | Total Samples | Total Files |
|-------|--------|--------|---------------|-------------|
| Good  | 5      | 5      | 10            | 30          |
| Okay  | 5      | 0      | 5             | 15          |
| Bad   | 5      | 0      | 5             | 15          |
| **Total** | **15** | **5** | **20** | **60** |

