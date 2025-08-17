## Manual Test Cases

This document outlines the manual test cases for the application's URL analysis workflow. The tests are designed to verify the correct classification of benign URLs and the accurate attribution of malicious URLs to their respective threat actor profiles.

### Test Case 1: Benign URL

**Objective**: To verify that a benign URL is correctly classified as harmless.

**Test Data**: A URL with features that align with the benign profile (e.g., valid SSL certificate, no IP address in URL, normal length, no suspicious characters).

#### Steps:

1. Run the application.

2. Input a URL with features of a benign website into the application.

3. Submit the URL for analysis.

#### Expected Result: The application's output should clearly state that the URL is Benign. The clustering model should not be engaged.

### Test Case 2: State-Sponsored URL

**Objective**: To verify that a malicious URL with state-sponsored characteristics is correctly classified and attributed.

**Test Data**: A URL with features of a state-sponsored actor (e.g., valid SSL certificate, deceptive subdomains, deceptive anchor links, no political keywords).

#### Steps:

1. Run the application.

2. Input a URL with state-sponsored features into the application.

3. Submit the URL for analysis.

#### Expected Result: 
The application's output should first classify the URL as Malicious and then attribute it to the State-Sponsored threat actor profile.

### Test Case 3: Organized Cybercrime URL

**Objective**: To verify that a malicious URL with organized cybercrime characteristics is correctly classified and attributed.

**Test Data**: A URL with features of organized cybercrime (e.g., use of an IP address, URL shortening service, no valid SSL, long and abnormal URL).

#### Steps:

1. Run the application.

2. Input a URL with organized cybercrime features into the application.

3. Submit the URL for analysis.

#### Expected Result: 
The application's output should first classify the URL as Malicious and then attribute it to the Organized Cybercrime threat actor profile.

### Test Case 4: Hacktivist URL

**Objective**: To verify that a malicious URL with hacktivist characteristics is correctly classified and attributed.

**Test Data**: A URL with features of a hacktivist (e.g., moderate use of IP address, average URL length, and a high likelihood of containing political keywords).

#### Steps:

1. Run the application.

2. Input a URL with hacktivist features into the application.

3. Submit the URL for analysis.

#### Expected Result: 
The application's output should first classify the URL as Malicious and then attribute it to the Hacktivist threat actor profile.