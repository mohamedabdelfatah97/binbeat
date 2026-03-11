# 1️⃣ Project Vision & Technical Stack Definition

## Project Vision (What We’re Building)

**AutoPulse HMI Lab** is a web-based automotive engineering project that simulates a modern **HMI (Human-Machine Interface)** system designed to model a realistic vehicle data pipeline.

The system focuses on:

- **OBD-style vehicle diagnostics**, including health monitoring and diagnostic trouble code (DTC) management  
- **Real-time performance telemetry visualization** (speed, RPM, throttle position, engine load, temperature, etc.)  
- **Realistic driving behavior modeling** using standardized driving cycles  
- **Synthetic OBD signal generation** layered on top of those cycles  
- **ML-based driver behavior classification** (Eco / Normal / Aggressive)  
- **Intelligent range estimation** based on driving style (planned predictive extension)

In simple terms:

> A modular automotive dashboard system that visualizes diagnostics and performance data, analyzes driving behavior using machine learning, and supports predictive vehicle performance estimation such as remaining range under specific driving styles.

This project is intentionally designed to go beyond a simple web dashboard. It aims to simulate a structured automotive data flow resembling real-world systems:

**Simulator → Backend Processing → Feature Extraction → ML Classification / Prediction → HMI Visualization**

The architecture is structured so that:

- Driver behavior classification is implemented first (supervised ML classification).
- Predictive regression (range estimation) is added as a natural extension.
- Each layer remains modular and independently extensible.

The overall objective is not merely functionality, but to build a system that is:

- Architecturally clean  
- Modular and extensible  
- Technically justified (machine learning applied where appropriate)  
- Professionally documented  
- Suitable for a strong GitHub portfolio presentation  

---

## Chosen Technology Stack

| Layer | Technology | Purpose |
|--------|------------|----------|
| **Backend** | Java (Spring Boot) | REST API, telemetry processing, ML inference |
| **Frontend** | Angular | HMI-style dashboard UI |
| **Simulator (Planned)** | Kotlin | Driving cycle modeling + synthetic OBD signal generation |
| **Database (Dev)** | H2 | Lightweight in-memory persistence during development |

---

## Why This Stack Was Chosen

### Backend — Java + Spring Boot

- Industry-standard framework for RESTful APIs  
- Embedded server for fast local development  
- Strong ecosystem (JPA, validation, actuator, etc.)  
- Clean architectural separation (controllers, services, repositories)  
- Suitable for future scaling or database upgrades  

### Frontend — Angular

- Structured framework ideal for medium-to-large applications  
- Strong typing and modular architecture  
- Built-in routing and state management patterns  
- Well-suited for building dashboard-style HMI interfaces  
- Adds professional weight to the portfolio  

### Simulator — Kotlin (Separate Module)

- Concise and expressive for data modeling  
- Fully compatible with JVM ecosystem  
- Clean separation between data generation and backend API logic  
- Adds architectural realism by mimicking a real telemetry producer  

### Database — H2 (Development Phase)

- Lightweight, in-memory database  
- Easy local setup with no external dependencies  
- Ideal for rapid iteration during early development  
- Can later be replaced with a persistent database if needed  

---

## High-Level Architecture Decision

The project follows a **monorepo structure with clearly separated modules**:

- `backend-spring/` → Spring Boot API layer  
- `frontend-angular/` → Angular HMI interface  
- `tools-simulator/` → Kotlin-based data generator (planned)  
- `data/` → Driving cycles and generated trip data (planned)  
- `docs/` → Technical documentation  
- `screenshots/` → UI previews for GitHub presentation  

### Intended Data Flow (Target Architecture)

Simulator  
→ Backend API (processing + feature extraction)  
→ ML classification / prediction  
→ Frontend dashboard visualization  

This layered architecture ensures:

- Clear responsibility boundaries  
- Maintainability and scalability  
- Realistic automotive-system modeling  
- Structured room for incremental feature expansion  

---

### Step 1 Status

Project vision defined.  
Technology stack selected.  
Architecture direction established.  
Scope includes both classification and predictive ML extension.

The foundation is now clearly defined before implementation begins.


# 2️⃣ Clean Machine Setup (Fresh MacBook Environment)

Day 1 began with a completely clean macOS environment.  
All development tools required for the backend, frontend, and future simulator modules were installed and configured from scratch.

The objective of this step was to establish a stable, reproducible development environment before writing any project code.

---

## 2.1 Homebrew Installation

Homebrew was installed as the primary package manager for macOS.

It allows controlled installation and version management of development tools such as Git and Node.js.

### Verification

```bash
brew --version
```

Homebrew’s binary directory (`/opt/homebrew/bin`) was added to the shell `PATH` to ensure installed tools take precedence over system defaults.

---

## 2.2 Git Installation & Configuration

macOS ships with a system Git version located at:

```bash
/usr/bin/git
```

To ensure consistency and access to newer versions, Git was installed via Homebrew.

### Verification

```bash
git --version
which git
```

The shell configuration file (`~/.zshrc`) was updated to prioritize Homebrew binaries:

```bash
export PATH="/opt/homebrew/bin:$PATH"
```

The shell was reloaded:

```bash
source ~/.zshrc
```

This ensured the correct Git installation was being used globally.

---

## 2.3 GitHub CLI Authentication

The GitHub CLI (`gh`) was used to authenticate the local machine with GitHub.

### Verification

```bash
gh auth status
```

Confirmed:

- Active GitHub account  
- Proper repository permissions  
- Secure token configuration  

This enables repository creation and remote interaction directly from the terminal.

---

## 2.4 Java Environment Setup (OpenJDK 17)

The backend requires Java 17.

### Verification of Installed JVM

```bash
/usr/libexec/java_home -V
java -version
```

The `JAVA_HOME` environment variable was configured inside `~/.zshrc`:

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PATH="$JAVA_HOME/bin:$PATH"
```

The shell was reloaded:

```bash
source ~/.zshrc
```

This ensures:

- Correct Java version is used  
- All Java-based tools (Gradle and later Kotlin) resolve properly  

---

## 2.5 Kotlin Readiness

No separate Kotlin installation was required at this stage.

Kotlin runs on the JVM and will use the configured Java 17 runtime when the simulator module is implemented.

This keeps the environment minimal while remaining future-ready.

---

## 2.6 Node.js & Angular Toolchain Setup

Since the frontend is built with Angular, Node.js and npm were required.

### Install Node Version Manager (nvm)

Configured inside `~/.zshrc`:

```bash
export NVM_DIR="$HOME/.nvm"
source "$(brew --prefix nvm)/nvm.sh"
```

Installed the latest LTS version of Node:

```bash
nvm install --lts
```

### Verification

```bash
node -v
npm -v
```

---

### Install Angular CLI

Angular CLI was installed globally:

```bash
npm install -g @angular/cli
```

### Verification

```bash
ng version
```

This provides project scaffolding, development server control, and build tooling for the frontend.

---

## 2.7 Visual Studio Code Setup

Visual Studio Code was installed as the primary development environment.

The following extensions were added:

- Java Extension Pack  
- Spring Boot Tools  
- Angular Language Service  

### Verification

- Integrated terminal recognizes Java  
- Git commands execute correctly  
- Node and Angular CLI available inside VS Code  

---

## Environment Validation

Final checks ensured all core tools were correctly configured:

```bash
java -version
git --version
node -v
npm -v
ng version
```

All tools resolved successfully through the configured `PATH`.

---

## Step 2 Outcome

At the end of this phase:

- Development environment fully configured  
- Backend toolchain ready (Java + Gradle)  
- Frontend toolchain ready (Node + Angular CLI)  
- GitHub authentication complete  
- IDE configured for multi-language development  

The system was now ready for project initialization and architecture setup.

# 3️⃣ Project Workspace & Version Control Initialization

With the development environment fully configured, the next step was to establish the project workspace and initialize version control.

This phase officially transformed the idea into a structured, version-controlled engineering project.

---

## 3.1 Create Development Workspace

A dedicated development directory was created to organize all projects:

```bash
mkdir -p ~/dev
cd ~/dev
```

Inside this directory, the main project folder was created:

```bash
mkdir autopulse-hmi-lab
cd autopulse-hmi-lab
```

At this stage, the project existed locally on the MacBook filesystem but was not yet version-controlled.

---

## 3.2 Define Initial Monorepo Structure

A monorepo structure was chosen to keep all modules under one repository while maintaining clean separation of responsibilities.

Initial folder structure:

```bash
mkdir -p backend-spring \
         frontend-angular \
         tools-simulator \
         data/cycles \
         data/trips \
         docs \
         screenshots
```

### Folder Purpose

- `backend-spring/` → Spring Boot backend API  
- `frontend-angular/` → Angular HMI application  
- `tools-simulator/` → Kotlin-based driving cycle simulator (planned)  
- `data/` → Real driving cycles and generated telemetry data (planned)  
- `docs/` → Technical documentation and development reports  
- `screenshots/` → UI previews for GitHub presentation  

This structure enforces modularity and future scalability from day one.

---

## 3.3 Initialize Git Repository

Git was initialized inside the root project directory:

```bash
git init
```

By default, Git created a `master` branch.  
To follow modern conventions, it was renamed to `main`:

```bash
git branch -m main
```

Verification:

```bash
git status
```

At this point:
- No commits existed
- All folders were untracked
- Repository was local only

---

## 3.4 Generate Spring Boot Backend Scaffold

The backend was bootstrapped using Spring Initializr via curl:

```bash
curl -L "https://start.spring.io/starter.zip?type=gradle-project&language=java&javaVersion=17&groupId=com.autopulse&artifactId=backend&name=backend&packageName=com.autopulse.backend&dependencies=web,validation,actuator,data-jpa,h2" -o backend.zip
```

The project was extracted into `backend-spring/`.

This generated:

- Gradle wrapper
- Embedded Tomcat configuration
- Spring Boot starter dependencies
- H2 in-memory database
- Actuator endpoints

The backend could now be started using:

```bash
cd backend-spring
./gradlew bootRun
```

Verification in browser:

```
http://localhost:8080/actuator/health
```

Response:

```json
{"status":"UP"}
```

This confirmed:
- Backend successfully running
- Embedded server operational
- Database initialized
- Spring Boot configured correctly

---

## 3.5 First Custom API Endpoint

A simple custom health endpoint was added:

```
GET /api/health
```

Verification:

```
http://localhost:8080/api/health
```

Response:

```json
{"service":"AutoPulse HMI Lab Backend","status":"ok"}
```

This confirmed:
- Controller layer functioning
- REST API reachable
- Application correctly wired

---

## 3.6 First Git Commit

All files were staged:

```bash
git add .
```

Initial commit:

```bash
git commit -m "feat: initial backend setup with Spring Boot and health endpoint"
```

Verification:

```bash
git log --oneline
```

This marked the official starting point of the project history.

---

## 3.7 Connect Repository to GitHub

A remote repository was created on GitHub.

The local repository was connected:

```bash
git remote add origin https://github.com/<username>/autopulse-hmi-lab.git
```

Pushed to GitHub:

```bash
git push -u origin main
```

The project now existed both locally and remotely.

Version control workflow established.

---

## Step 3 Outcome

At the end of this phase:

- Project workspace structured
- Monorepo architecture defined
- Git repository initialized
- Backend scaffold generated
- First REST endpoint implemented
- Initial commit created
- Repository successfully pushed to GitHub

The project officially transitioned from concept to a structured, version-controlled engineering system.

# 4️⃣ Backend Bootstrapping (Spring Boot)

With the project workspace and Git repository initialized, the next step was to generate and run the backend application using Spring Boot.

This marked the beginning of the actual backend implementation.

---

## 4.1 Generate Spring Boot Project

The backend was generated using Spring Initializr via `curl`.

Command executed from the root project directory:

```bash
curl -L "https://start.spring.io/starter.zip?type=gradle-project&language=java&javaVersion=17&groupId=com.autopulse&artifactId=backend&name=backend&packageName=com.autopulse.backend&dependencies=web,validation,actuator,data-jpa,h2" -o backend.zip
```

The generated project was extracted and placed inside:

```
backend-spring/
```

This automatically created:

- Gradle wrapper
- Embedded Tomcat configuration
- Spring Boot starter dependencies
- H2 in-memory database
- Actuator monitoring endpoints

---

## 4.2 Inspect Generated Structure

The generated backend structure:

```
backend-spring/
 ├── build.gradle
 ├── gradlew
 ├── settings.gradle
 └── src/main/java/com/autopulse/backend/
     └── BackendApplication.java
```

The main entry point:

```
BackendApplication.java
```

This file contains the `@SpringBootApplication` annotation and starts the application.

---

## 4.3 Run the Backend for the First Time

From inside the backend directory:

```bash
cd backend-spring
./gradlew bootRun
```

The first run triggered:

- Dependency download via Gradle
- Project compilation
- Embedded Tomcat startup
- H2 database initialization

---

## 4.4 Verify Successful Startup

From the console logs:

```
Tomcat started on port 8080 (http)
Started BackendApplication
```

This confirmed:

- Embedded web server running
- Application successfully bootstrapped
- Port 8080 bound correctly
- No critical startup errors

---

## 4.5 Verify Actuator Health Endpoint

Since Actuator dependency was included, the default health endpoint was available.

Browser verification:

```
http://localhost:8080/actuator/health
```

Response:

```json
{"status":"UP"}
```

This confirmed:

- Application running correctly
- Internal health checks passing
- Spring Boot environment properly configured

---

## Step 4 Outcome

At the end of this phase:

- Spring Boot backend successfully generated
- Embedded Tomcat server running
- H2 in-memory database initialized
- Actuator health endpoint verified
- Backend reachable via browser on port 8080

The backend infrastructure was now fully bootstrapped and operational.

# 5️⃣ Backend Verification & Custom API Creation

With the backend successfully bootstrapped and infrastructure confirmed operational (Step 4), the next objective was to implement and verify a custom REST endpoint.

This marks the transition from “framework running” to “application logic implemented”.

---

## 5.1 Create Custom REST Controller

A new controller class was created under:

```
backend-spring/src/main/java/com/autopulse/backend/api/
```

File created:

```
HealthController.java
```

This controller exposes a simple API endpoint:

- HTTP Method: `GET`
- Route: `/api/health`
- Response: JSON object containing service name and status

The purpose of this endpoint is to:

- Confirm routing works correctly
- Validate JSON serialization
- Provide a stable endpoint for frontend integration

---

## 5.2 Run the Backend

From inside the backend directory:

```bash
cd ~/dev/autopulse-hmi-lab/backend-spring
./gradlew bootRun
```

Confirmed successful startup in console logs:

```
Tomcat started on port 8080 (http)
Started BackendApplication
```

---

## 5.3 Verify Custom API Endpoint

Tested in browser:

```
http://localhost:8080/api/health
```

Expected response:

```json
{
  "service": "AutoPulse HMI Lab Backend",
  "status": "ok"
}
```

This confirmed:

- Spring correctly maps `/api/health`
- Controller method executes successfully
- Java response automatically converted to JSON
- HTTP request-response cycle functioning correctly

---

## Step 5 Outcome

At the end of this phase:

- First custom REST endpoint implemented
- JSON serialization confirmed
- Backend now exposes a usable API route
- Application logic layer successfully verified

The backend is no longer just running — it now actively serves application-level data.

# 6️⃣ First Git Commit & Remote Repository Setup

After successfully bootstrapping the backend and implementing the first custom API endpoint, the next step was to formally track the project using Git and connect it to a remote GitHub repository.

This marks the official beginning of the project's version-controlled history.

---

## 6.1 Stage Project Files

From the root project directory:

```bash
cd ~/dev/autopulse-hmi-lab
git status
```

At this stage:

- `backend-spring/` appeared as untracked
- No commits existed yet
- Repository had been initialized but not recorded

All files were staged:

```bash
git add .
```

---

## 6.2 Create Initial Commit

The first commit was created with a descriptive message:

```bash
git commit -m "feat: initial backend setup with Spring Boot and health endpoint"
```

Verification:

```bash
git log --oneline
```

This confirmed:

- Commit successfully created
- Backend scaffold and controller now tracked
- Project history officially started

---

## 6.3 Connect to GitHub Remote Repository

A remote repository was created on GitHub.

The local repository was connected to the remote:

```bash
git remote add origin https://github.com/<username>/autopulse-hmi-lab.git
```

Verification:

```bash
git remote -v
```

---

## 6.4 Push to Remote Repository

The project was pushed to GitHub:

```bash
git push -u origin main
```

The `-u` flag sets the upstream branch, allowing future pushes with:

```bash
git push
```

This confirmed:

- Remote repository successfully connected
- Initial commit visible on GitHub
- Local and remote branches synchronized

---

## Step 6 Outcome

At the end of this phase:

- First commit created
- Backend scaffold version-controlled
- Remote GitHub repository connected
- Initial code pushed successfully
- Project officially live on GitHub

AutoPulse HMI Lab transitioned from a local experiment into a publicly tracked engineering project.


# 7️⃣ Angular Application Bootstrapping

With the backend implemented, verified, and pushed to GitHub, the next phase was to initialize the frontend application using Angular.

This step established the UI foundation of the project but did not yet integrate it with the backend.

---

## 7.1 Create Angular Application

From inside the frontend directory:

```bash
cd ~/dev/autopulse-hmi-lab/frontend-angular
ng new autopulse-ui --routing --style=scss
```

During setup, the following options were selected:

- Routing: ✅ Enabled  
- Stylesheet format: SCSS  
- Server-Side Rendering (SSR): ✅ Enabled  
- Angular analytics: ❌ Disabled  

This generated a fully structured Angular project inside:

```
frontend-angular/autopulse-ui/
```

---

## 7.2 Inspect Generated Structure

The Angular CLI created a standard project structure:

```
autopulse-ui/
 ├── src/
 │   ├── app/
 │   ├── main.ts
 │   └── index.html
 ├── angular.json
 ├── package.json
 ├── tsconfig.json
 └── .vscode/
```

Key components:

- `src/app/` → Main application logic
- `angular.json` → Build configuration
- `package.json` → Dependency definitions
- `main.ts` → Application entry point

---

## 7.3 Install Dependencies

After generation, npm automatically installed all required dependencies:

- Angular core packages
- TypeScript
- Vite (build tool)
- Development tooling

The console confirmed:

```
✔ Packages installed successfully.
```

---

## 7.4 Start Angular Development Server

From inside the Angular project directory:

```bash
cd autopulse-ui
ng serve
```

Console output:

```
Local: http://localhost:4200/
Watch mode enabled.
```

Opening the browser confirmed:

- Angular application running successfully
- Default template displayed
- Development server operational
- Hot reload active

At this stage:

- Backend running on port `8080`
- Frontend running independently on port `4200`

The systems were not yet connected.

---

## Step 7 Outcome

At the end of this phase:

- Angular project successfully generated
- Dependencies installed
- Development server running
- Frontend application accessible in browser
- Foundation ready for backend integration

AutoPulse HMI Lab now had both backend and frontend layers initialized.


# 8️⃣ Full-Stack Integration (Frontend ↔ Backend)

With both backend and frontend running independently, the next step was to connect them into a functioning full-stack system.

At this stage:

- Backend running on `http://localhost:8080`
- Frontend running on `http://localhost:4200`

Since both run on different ports, integration required proper routing and development proxy configuration.

---

## 8.1 Identify Cross-Origin Setup

Because:

- Angular runs on port `4200`
- Spring Boot runs on port `8080`

Requests from frontend to backend would normally trigger CORS restrictions.

To simplify development, a proxy configuration was implemented in Angular.

---

## 8.2 Create Angular Proxy Configuration

Inside the Angular project root:

```
frontend-angular/autopulse-ui/
```

A file was created:

```
proxy.conf.json
```

Content:

```json
{
  "/api": {
    "target": "http://localhost:8080",
    "secure": false,
    "changeOrigin": true
  }
}
```

This configuration instructs Angular:

- Any request starting with `/api`
- Forward it to `http://localhost:8080`
- Avoid CORS issues during development

---

## 8.3 Start Angular with Proxy Enabled

The development server was restarted using:

```bash
ng serve --proxy-config proxy.conf.json
```

Console confirmed:

```
Local: http://localhost:4200/
```

Proxy now active.

---

## 8.4 Implement Frontend Health Check Request

To verify integration, the frontend was modified to request the backend endpoint:

```
GET /api/health
```

A button was added in the UI to trigger the request, and the response was displayed on screen.

When clicked:

1. Angular sends request to `/api/health`
2. Proxy forwards request to backend (`localhost:8080`)
3. Spring Boot processes request
4. JSON response returned
5. Angular displays result

---

## 8.5 Verify Successful Integration

Browser verification:

```
http://localhost:4200/api/health
```

Response received:

```json
{
  "service": "AutoPulse HMI Lab Backend",
  "status": "ok"
}
```

Additionally, the UI successfully displayed the backend response after button interaction.

This confirmed:

- Proxy functioning correctly
- Backend reachable from frontend
- HTTP communication working
- JSON parsing operational
- Full request-response cycle complete

---

## Full-Stack Execution Flow

```
User Interaction (Frontend UI)
        ↓
Angular HTTP Request (/api/health)
        ↓
Angular Proxy
        ↓
Spring Boot Controller
        ↓
JSON Response
        ↓
Angular UI Update
```

The system now operates as a connected full-stack architecture.

---

## Step 8 Outcome

At the end of this phase:

- Angular proxy configured
- Backend successfully reachable from frontend
- First full-stack API call verified
- CORS avoided using development proxy
- Client-server communication confirmed

AutoPulse HMI Lab officially became a functional full-stack application.


# 9️⃣ Debugging & Issue Resolution

During Day 1, multiple environment and runtime issues were encountered and resolved.  
This phase reflects practical problem-solving and system-level understanding rather than feature implementation.

Documenting these issues demonstrates engineering maturity and environment awareness.

---

## 9.1 Homebrew Git vs System Git Conflict

### Problem

After installing Git via Homebrew, the system was still using:

```
/usr/bin/git
```

instead of the Homebrew version located in:

```
/opt/homebrew/bin/git
```

### Diagnosis

Command used:

```bash
which git
```

Result indicated the system Git was taking precedence in the `PATH`.

### Resolution

Updated `~/.zshrc`:

```bash
export PATH="/opt/homebrew/bin:$PATH"
```

Reloaded shell:

```bash
source ~/.zshrc
```

Verification:

```bash
which git
git --version
```

Confirmed Homebrew Git version now active.

---

## 9.2 JAVA_HOME Environment Configuration

### Problem

Although Java 17 was installed, environment variables were not explicitly set, which could cause issues with build tools.

### Resolution

Configured `JAVA_HOME` inside `~/.zshrc`:

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PATH="$JAVA_HOME/bin:$PATH"
```

Reloaded shell:

```bash
source ~/.zshrc
```

Verification:

```bash
echo $JAVA_HOME
java -version
```

Confirmed correct Java version in use.

---

## 9.3 Gradle Execution Confusion

### Problem

Attempted to execute:

```bash
./gradlew bootRun
```

from the wrong directory (`autopulse-ui`), resulting in:

```
zsh: no such file or directory: ./gradlew
```

### Cause

Gradle wrapper exists only inside:

```
backend-spring/
```

### Resolution

Navigated to correct directory:

```bash
cd ~/dev/autopulse-hmi-lab/backend-spring
./gradlew bootRun
```

Backend started successfully.

---

## 9.4 Angular SSR / Vite Cache Error

### Problem

After modifying Angular files, the development server produced an error:

```
There is a new version of the pre-bundle...
```

This was related to Vite’s SSR caching system.

### Resolution

Stopped Angular server using:

```bash
Ctrl + C
```

Restarted with:

```bash
ng serve --proxy-config proxy.conf.json
```

The cache rebuilt automatically and the issue was resolved.

---

## 9.5 Understanding Development Server States

At times, it appeared that the backend or frontend was "not working" when:

- The server was not running
- The wrong port was accessed
- The wrong directory was used
- The development server had been terminated

This reinforced the importance of verifying:

- Which process is running
- Which port is being used
- Current working directory (`pwd`)
- Console output for errors

---

## 9.6 Permission Confusion When Running Directory as Command

An attempt was made to execute:

```bash
/Users/marcelo/dev/autopulse-hmi-lab/backend-spring
```

Result:

```
zsh: permission denied
```

Cause:

- Attempted to execute a directory instead of navigating into it.

Resolution:

```bash
cd backend-spring
```

Then run:

```bash
./gradlew bootRun
```

---

## Step 9 Outcome

At the end of this phase:

- PATH precedence properly configured
- JAVA_HOME correctly set
- Git version conflict resolved
- Gradle execution path understood
- Angular SSR cache issue fixed
- Stronger understanding of terminal workflow established

This step significantly improved command-line confidence and environment debugging skills.
# 🔟 Final Architecture, Concepts Learned & Day 1 Assessment

## Final Working Architecture Overview

By the end of Day 1, AutoPulse HMI Lab consisted of a functioning full-stack system:

```
Angular Frontend (Port 4200)
        ↓
Development Proxy
        ↓
Spring Boot Backend (Port 8080)
        ↓
H2 In-Memory Database
```

### Active Components

- Spring Boot backend running with embedded Tomcat  
- Custom REST endpoint: `/api/health`  
- Angular frontend running with Vite-based dev server  
- Proxy configuration forwarding `/api` calls to backend  
- Full request-response cycle verified  

The system is now a functional client-server architecture ready for telemetry modeling and ML feature expansion.

---

## Technical Concepts Learned

Day 1 established foundational knowledge in:

### Backend Concepts
- Spring Boot application lifecycle  
- Embedded Tomcat server behavior  
- REST controller structure  
- JSON serialization (Java → HTTP response)  
- Actuator health monitoring  
- Gradle wrapper usage  

### Frontend Concepts
- Angular project structure  
- Development server workflow  
- Proxy configuration for CORS handling  
- HTTP request handling in Angular  
- Client-server communication flow  

### Environment & Tooling
- PATH precedence management (Homebrew vs system binaries)  
- JAVA_HOME configuration  
- Git version control workflow  
- Git remote setup & upstream tracking  
- Debugging terminal and directory issues  

---

## End-of-Day Status Assessment

At the conclusion of Day 1:

✅ Development environment fully configured  
✅ Backend bootstrapped and verified  
✅ Custom API endpoint implemented  
✅ First commit pushed to GitHub  
✅ Angular frontend initialized  
✅ Full-stack communication successfully established  
✅ Multiple environment issues resolved  

### System State

The project has transitioned from concept to:

> A working, version-controlled, full-stack automotive dashboard foundation.

The infrastructure is now stable and ready for:

- Telemetry data modeling  
- Simulator module implementation (Kotlin)  
- UI component modularization  
- ML feature integration  

Day 1 successfully established a clean technical foundation for all future development.