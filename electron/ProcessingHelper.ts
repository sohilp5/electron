// electron/ProcessingHelper.ts
import fs from "node:fs"
import path from "node:path"
import { ScreenshotHelper } from "./ScreenshotHelper"
import { IProcessingHelperDeps } from "./main"
import * as axios from "axios"
import { app, BrowserWindow, dialog } from "electron"
import { OpenAI } from "openai"
import { configHelper } from "./ConfigHelper"
import Anthropic from '@anthropic-ai/sdk';

// Interface for Gemini API requests
interface GeminiMessage {
  role: string;
  parts: Array<{
    text?: string;
    inlineData?: {
      mimeType: string;
      data: string;
    }
  }>;
}

interface GeminiResponse {
  candidates: Array<{
    content: {
      parts: Array<{
        text: string;
      }>;
    };
    finishReason: string;
  }>;
}
interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: Array<{
    type: 'text' | 'image';
    text?: string;
    source?: {
      type: 'base64';
      media_type: string;
      data: string;
    };
  }>;
}
export class ProcessingHelper {
  private deps: IProcessingHelperDeps
  private screenshotHelper: ScreenshotHelper
  private openaiClient: OpenAI | null = null
  private geminiApiKey: string | null = null
  private anthropicClient: Anthropic | null = null

  // AbortControllers for API requests
  private currentProcessingAbortController: AbortController | null = null
  private currentExtraProcessingAbortController: AbortController | null = null

  constructor(deps: IProcessingHelperDeps) {
    this.deps = deps
    this.screenshotHelper = deps.getScreenshotHelper()
    
    // Initialize AI client based on config
    this.initializeAIClient();
    
    // Listen for config changes to re-initialize the AI client
    configHelper.on('config-updated', () => {
      this.initializeAIClient();
    });
  }
  
  /**
   * Initialize or reinitialize the AI client with current config
   */
  private initializeAIClient(): void {
    try {
      const config = configHelper.loadConfig();
      
      if (config.apiProvider === "openai") {
        if (config.apiKey) {
          this.openaiClient = new OpenAI({ 
            apiKey: config.apiKey,
            timeout: 60000, // 60 second timeout
            maxRetries: 2   // Retry up to 2 times
          });
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.log("OpenAI client initialized successfully");
        } else {
          this.openaiClient = null;
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.warn("No API key available, OpenAI client not initialized");
        }
      } else if (config.apiProvider === "gemini"){
        // Gemini client initialization
        this.openaiClient = null;
        this.anthropicClient = null;
        if (config.apiKey) {
          this.geminiApiKey = config.apiKey;
          console.log("Gemini API key set successfully");
        } else {
          this.openaiClient = null;
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.warn("No API key available, Gemini client not initialized");
        }
      } else if (config.apiProvider === "anthropic") {
        // Reset other clients
        this.openaiClient = null;
        this.geminiApiKey = null;
        if (config.apiKey) {
          this.anthropicClient = new Anthropic({
            apiKey: config.apiKey,
            timeout: 60000,
            maxRetries: 2
          });
          console.log("Anthropic client initialized successfully");
        } else {
          this.openaiClient = null;
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.warn("No API key available, Anthropic client not initialized");
        }
      }
    } catch (error) {
      console.error("Failed to initialize AI client:", error);
      this.openaiClient = null;
      this.geminiApiKey = null;
      this.anthropicClient = null;
    }
  }

  private async waitForInitialization(
    mainWindow: BrowserWindow
  ): Promise<void> {
    let attempts = 0
    const maxAttempts = 50 // 5 seconds total

    while (attempts < maxAttempts) {
      const isInitialized = await mainWindow.webContents.executeJavaScript(
        "window.__IS_INITIALIZED__"
      )
      if (isInitialized) return
      await new Promise((resolve) => setTimeout(resolve, 100))
      attempts++
    }
    throw new Error("App failed to initialize after 5 seconds")
  }

  private async getCredits(): Promise<number> {
    const mainWindow = this.deps.getMainWindow()
    if (!mainWindow) return 999 // Unlimited credits in this version

    try {
      await this.waitForInitialization(mainWindow)
      return 999 // Always return sufficient credits to work
    } catch (error) {
      console.error("Error getting credits:", error)
      return 999 // Unlimited credits as fallback
    }
  }

  private async getLanguage(): Promise<string> {
    try {
      // Get language from config
      const config = configHelper.loadConfig();
      if (config.language) {
        return config.language;
      }
      
      // Fallback to window variable if config doesn't have language
      const mainWindow = this.deps.getMainWindow()
      if (mainWindow) {
        try {
          await this.waitForInitialization(mainWindow)
          const language = await mainWindow.webContents.executeJavaScript(
            "window.__LANGUAGE__"
          )

          if (
            typeof language === "string" &&
            language !== undefined &&
            language !== null
          ) {
            return language;
          }
        } catch (err) {
          console.warn("Could not get language from window", err);
        }
      }
      
      // Default fallback
      return "python";
    } catch (error) {
      console.error("Error getting language:", error)
      return "python"
    }
  }

  public async processScreenshots(): Promise<void> {
    const mainWindow = this.deps.getMainWindow()
    if (!mainWindow) return

    const config = configHelper.loadConfig();
    
    // First verify we have a valid AI client
    if (config.apiProvider === "openai" && !this.openaiClient) {
      this.initializeAIClient();
      
      if (!this.openaiClient) {
        console.error("OpenAI client not initialized");
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.API_KEY_INVALID
        );
        return;
      }
    } else if (config.apiProvider === "gemini" && !this.geminiApiKey) {
      this.initializeAIClient();
      
      if (!this.geminiApiKey) {
        console.error("Gemini API key not initialized");
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.API_KEY_INVALID
        );
        return;
      }
    } else if (config.apiProvider === "anthropic" && !this.anthropicClient) {
      // Add check for Anthropic client
      this.initializeAIClient();
      
      if (!this.anthropicClient) {
        console.error("Anthropic client not initialized");
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.API_KEY_INVALID
        );
        return;
      }
    }

    const view = this.deps.getView()
    console.log("Processing screenshots in view:", view)

    if (view === "queue") {
      mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.INITIAL_START)
      const screenshotQueue = this.screenshotHelper.getScreenshotQueue()
      console.log("Processing main queue screenshots:", screenshotQueue)
      
      // Check if the queue is empty
      if (!screenshotQueue || screenshotQueue.length === 0) {
        console.log("No screenshots found in queue");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }

      // Check that files actually exist
      const existingScreenshots = screenshotQueue.filter(path => fs.existsSync(path));
      if (existingScreenshots.length === 0) {
        console.log("Screenshot files don't exist on disk");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }

      try {
        // Initialize AbortController
        this.currentProcessingAbortController = new AbortController()
        const { signal } = this.currentProcessingAbortController

        const screenshots = await Promise.all(
          existingScreenshots.map(async (path) => {
            try {
              return {
                path,
                preview: await this.screenshotHelper.getImagePreview(path),
                data: fs.readFileSync(path).toString('base64')
              };
            } catch (err) {
              console.error(`Error reading screenshot ${path}:`, err);
              return null;
            }
          })
        )

        // Filter out any nulls from failed screenshots
        const validScreenshots = screenshots.filter(Boolean);
        
        if (validScreenshots.length === 0) {
          throw new Error("Failed to load screenshot data");
        }

        const result = await this.processScreenshotsHelper(validScreenshots, signal)

        if (!result.success) {
          console.log("Processing failed:", result.error)
          if (result.error?.includes("API Key") || result.error?.includes("OpenAI") || result.error?.includes("Gemini")) {
            mainWindow.webContents.send(
              this.deps.PROCESSING_EVENTS.API_KEY_INVALID
            )
          } else {
            mainWindow.webContents.send(
              this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR,
              result.error
            )
          }
          // Reset view back to queue on error
          console.log("Resetting view to queue due to error")
          this.deps.setView("queue")
          return
        }

        // Only set view to solutions if processing succeeded
        console.log("Setting view to solutions after successful processing")
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.SOLUTION_SUCCESS,
          result.data
        )
        this.deps.setView("solutions")
      } catch (error: any) {
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR,
          error
        )
        console.error("Processing error:", error)
        if (axios.isCancel(error)) {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR,
            "Processing was canceled by the user."
          )
        } else {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR,
            error.message || "Server error. Please try again."
          )
        }
        // Reset view back to queue on error
        console.log("Resetting view to queue due to error")
        this.deps.setView("queue")
      } finally {
        this.currentProcessingAbortController = null
      }
    } else {
      // view == 'solutions'
      const extraScreenshotQueue =
        this.screenshotHelper.getExtraScreenshotQueue()
      console.log("Processing extra queue screenshots:", extraScreenshotQueue)
      
      // Check if the extra queue is empty
      if (!extraScreenshotQueue || extraScreenshotQueue.length === 0) {
        console.log("No extra screenshots found in queue");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        
        return;
      }

      // Check that files actually exist
      const existingExtraScreenshots = extraScreenshotQueue.filter(path => fs.existsSync(path));
      if (existingExtraScreenshots.length === 0) {
        console.log("Extra screenshot files don't exist on disk");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }
      
      mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.DEBUG_START)

      // Initialize AbortController
      this.currentExtraProcessingAbortController = new AbortController()
      const { signal } = this.currentExtraProcessingAbortController

      try {
        // Get all screenshots (both main and extra) for processing
        const allPaths = [
          ...this.screenshotHelper.getScreenshotQueue(),
          ...existingExtraScreenshots
        ];
        
        const screenshots = await Promise.all(
          allPaths.map(async (path) => {
            try {
              if (!fs.existsSync(path)) {
                console.warn(`Screenshot file does not exist: ${path}`);
                return null;
              }
              
              return {
                path,
                preview: await this.screenshotHelper.getImagePreview(path),
                data: fs.readFileSync(path).toString('base64')
              };
            } catch (err) {
              console.error(`Error reading screenshot ${path}:`, err);
              return null;
            }
          })
        )
        
        // Filter out any nulls from failed screenshots
        const validScreenshots = screenshots.filter(Boolean);
        
        if (validScreenshots.length === 0) {
          throw new Error("Failed to load screenshot data for debugging");
        }
        
        console.log(
          "Combined screenshots for processing:",
          validScreenshots.map((s) => s.path)
        )

        const result = await this.processExtraScreenshotsHelper(
          validScreenshots,
          signal
        )

        if (result.success) {
          this.deps.setHasDebugged(true)
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.DEBUG_SUCCESS,
            result.data
          )
        } else {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.DEBUG_ERROR,
            result.error
          )
        }
      } catch (error: any) {
        if (axios.isCancel(error)) {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.DEBUG_ERROR,
            "Extra processing was canceled by the user."
          )
        } else {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.DEBUG_ERROR,
            error.message
          )
        }
      } finally {
        this.currentExtraProcessingAbortController = null
      }
    }
  }

  private async processScreenshotsHelper(
    screenshots: Array<{ path: string; data: string }>,
    signal: AbortSignal
  ) {
    try {
      const config = configHelper.loadConfig();
      const language = await this.getLanguage();
      const mainWindow = this.deps.getMainWindow();
      
      // Step 1: Extract problem info using AI Vision API (OpenAI or Gemini)
      const imageDataList = screenshots.map(screenshot => screenshot.data);
      
      // Update the user on progress
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Analyzing problem from screenshots...",
          progress: 20
        });
      }

      let problemInfo;
      
      if (config.apiProvider === "openai") {
        // Verify OpenAI client
        if (!this.openaiClient) {
          this.initializeAIClient(); // Try to reinitialize
          
          if (!this.openaiClient) {
            return {
              success: false,
              error: "OpenAI API key not configured or invalid. Please check your settings."
            };
          }
        }

        // Use OpenAI for processing
        const messages = [
          {
            role: "system" as const, 
            content: "You are a coding challenge interpreter. Analyze the screenshot of the coding problem and extract all relevant information. Return the information in JSON format with these fields: problem_statement, constraints, example_input, example_output. Just return the structured JSON without any other text."
          },
          {
            role: "user" as const,
            content: [
              {
                type: "text" as const, 
                text: `Extract the coding problem details from these screenshots. Return in JSON format. Preferred coding language we gonna use for this problem is ${language}.`
              },
              ...imageDataList.map(data => ({
                type: "image_url" as const,
                image_url: { url: `data:image/png;base64,${data}` }
              }))
            ]
          }
        ];

        // Send to OpenAI Vision API
        const extractionResponse = await this.openaiClient.chat.completions.create({
          model: config.extractionModel || "gpt-4o",
          messages: messages,
          max_tokens: 4000,
          temperature: 0.2
        });

        // Parse the response
        try {
          const responseText = extractionResponse.choices[0].message.content;
          // Handle when OpenAI might wrap the JSON in markdown code blocks
          const jsonText = responseText.replace(/```json|```/g, '').trim();
          problemInfo = JSON.parse(jsonText);
        } catch (error) {
          console.error("Error parsing OpenAI response:", error);
          return {
            success: false,
            error: "Failed to parse problem information. Please try again or use clearer screenshots."
          };
        }
      } else if (config.apiProvider === "gemini")  {
        // Use Gemini API
        if (!this.geminiApiKey) {
          return {
            success: false,
            error: "Gemini API key not configured. Please check your settings."
          };
        }

        try {
          // Create Gemini message structure
          const geminiMessages: GeminiMessage[] = [
            {
              role: "user",
              parts: [
                {
                  text: `You are a coding challenge interpreter. Analyze the screenshots of the coding problem and extract all relevant information. Return the information in JSON format with these fields: problem_statement, constraints, example_input, example_output. Just return the structured JSON without any other text. Preferred coding language we gonna use for this problem is ${language}.`
                },
                ...imageDataList.map(data => ({
                  inlineData: {
                    mimeType: "image/png",
                    data: data
                  }
                }))
              ]
            }
          ];

          // Make API request to Gemini
          const response = await axios.default.post(
            `https://generativelanguage.googleapis.com/v1beta/models/${config.extractionModel || "gemini-2.0-flash"}:generateContent?key=${this.geminiApiKey}`,
            {
              contents: geminiMessages,
              generationConfig: {
                temperature: 0.2,
                maxOutputTokens: 4000
              }
            },
            { signal }
          );

          const responseData = response.data as GeminiResponse;
          
          if (!responseData.candidates || responseData.candidates.length === 0) {
            throw new Error("Empty response from Gemini API");
          }
          
          const responseText = responseData.candidates[0].content.parts[0].text;
          
          // Handle when Gemini might wrap the JSON in markdown code blocks
          const jsonText = responseText.replace(/```json|```/g, '').trim();
          problemInfo = JSON.parse(jsonText);
        } catch (error) {
          console.error("Error using Gemini API:", error);
          return {
            success: false,
            error: "Failed to process with Gemini API. Please check your API key or try again later."
          };
        }
      } else if (config.apiProvider === "anthropic") {
        if (!this.anthropicClient) {
          return {
            success: false,
            error: "Anthropic API key not configured. Please check your settings."
          };
        }

        try {
          const messages = [
            {
              role: "user" as const,
              content: [
                {
                  type: "text" as const,
                  text: `Extract the coding problem details from these screenshots. Return in JSON format with these fields: problem_statement, constraints, example_input, example_output. Preferred coding language is ${language}.`
                },
                ...imageDataList.map(data => ({
                  type: "image" as const,
                  source: {
                    type: "base64" as const,
                    media_type: "image/png" as const,
                    data: data
                  }
                }))
              ]
            }
          ];

          const response = await this.anthropicClient.messages.create({
            model: config.extractionModel || "claude-3-7-sonnet-20250219",
            max_tokens: 4000,
            messages: messages,
            temperature: 0.2
          });

          const responseText = (response.content[0] as { type: 'text', text: string }).text;
          const jsonText = responseText.replace(/```json|```/g, '').trim();
          problemInfo = JSON.parse(jsonText);
        } catch (error: any) {
          console.error("Error using Anthropic API:", error);

          // Add specific handling for Claude's limitations
          if (error.status === 429) {
            return {
              success: false,
              error: "Claude API rate limit exceeded. Please wait a few minutes before trying again."
            };
          } else if (error.status === 413 || (error.message && error.message.includes("token"))) {
            return {
              success: false,
              error: "Your screenshots contain too much information for Claude to process. Switch to OpenAI or Gemini in settings which can handle larger inputs."
            };
          }

          return {
            success: false,
            error: "Failed to process with Anthropic API. Please check your API key or try again later."
          };
        }
      }
      
      // Update the user on progress
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Problem analyzed successfully. Preparing to generate solution...",
          progress: 40
        });
      }

      // Store problem info in AppState
      this.deps.setProblemInfo(problemInfo);

      // Send first success event
      if (mainWindow) {
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.PROBLEM_EXTRACTED,
          problemInfo
        );

        // Generate solutions after successful extraction
        const solutionsResult = await this.generateSolutionsHelper(signal);
        if (solutionsResult.success) {
          // Clear any existing extra screenshots before transitioning to solutions view
          this.screenshotHelper.clearExtraScreenshotQueue();
          
          // Final progress update
          mainWindow.webContents.send("processing-status", {
            message: "Solution generated successfully",
            progress: 100
          });
          
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.SOLUTION_SUCCESS,
            solutionsResult.data
          );
          return { success: true, data: solutionsResult.data };
        } else {
          throw new Error(
            solutionsResult.error || "Failed to generate solutions"
          );
        }
      }

      return { success: false, error: "Failed to process screenshots" };
    } catch (error: any) {
      // If the request was cancelled, don't retry
      if (axios.isCancel(error)) {
        return {
          success: false,
          error: "Processing was canceled by the user."
        };
      }
      
      // Handle OpenAI API errors specifically
      if (error?.response?.status === 401) {
        return {
          success: false,
          error: "Invalid OpenAI API key. Please check your settings."
        };
      } else if (error?.response?.status === 429) {
        return {
          success: false,
          error: "API rate limit exceeded or insufficient credits. Please try again later."
        };
      } else if (error?.response?.status === 500) {
        return {
          success: false,
          error: "OpenAI server error. Please try again later."
        };
      }

      console.error("API Error Details:", error);
      return { 
        success: false, 
        error: error.message || "Failed to process screenshots. Please try again." 
      };
    }
  }

  private async generateSolutionsHelper(signal: AbortSignal) {
    try {
      const problemInfo = this.deps.getProblemInfo();
      const language = await this.getLanguage();
      const config = configHelper.loadConfig();
      const mainWindow = this.deps.getMainWindow();

      if (!problemInfo) {
        throw new Error("No problem info available");
      }
      
      // == STAGE 1: THE "PLAN" ==
      const planPrompt = `
You are an expert coding interview coach. Your task is to create a high-level plan to solve the following problem. Do not write the final optimal code yet. Instead, walk through the thought process a candidate should follow.

The response MUST be structured with the following numbered sections:

1.  **Initial Analysis & Questions:**
    * Start with a one-sentence summary of the goal.
    * List 2-3 essential clarifying questions about inputs, outputs, or edge cases.

2.  **Brute-Force Approach:**
    * Briefly explain the logic of the most straightforward, often naive, solution.
    * State its Time and Space Complexity and explain its primary inefficiency or bottleneck.

3.  **The "Aha!" Moment / Optimization:**
    * Concisely explain the key insight that leads to a better solution. (e.g., "The inefficiency comes from repeated lookups. We can optimize this by using a hash map for O(1) lookups.")

4.  **Optimal Solution Plan & Example Walkthrough:**
    * First, outline the high-level steps for the optimal algorithm.
    * Next, walk through these steps using the provided example: \`${problemInfo.example_input || "No example provided, please create a simple one (e.g., for Two Sum, use `nums = [2, 7, 11, 15], target = 9`)"}\`.
    * Show how key variables or data structures (like arrays, pointers, hashmaps) change during the execution of the algorithm with this example. This should be a step-by-step trace of the logic.

---
**PROBLEM DETAILS:**

**PROBLEM STATEMENT:**
${problemInfo.problem_statement}

**CONSTRAINTS:**
${problemInfo.constraints || "No specific constraints provided."}

**EXAMPLE INPUT:**
${problemInfo.example_input || "Not provided."}

**LANGUAGE:** ${language}
---
`;

      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Developing solution strategy and plan...",
          progress: 60
        });
      }

      let planResponse = "";
      if (config.apiProvider === "openai") {
        if (!this.openaiClient) return { success: false, error: "OpenAI Client not initialized." };
        const response = await this.openaiClient.chat.completions.create({
          model: config.solutionModel || "gpt-4o",
          messages: [{ role: "system", content: "You are an expert coding interview coach." }, { role: "user", content: planPrompt }],
          max_tokens: 4000,
          temperature: 0.2
        });
        planResponse = response.choices[0].message.content || "";
      } else if (config.apiProvider === "gemini") {
        if (!this.geminiApiKey) return { success: false, error: "Gemini API Key not configured." };
        const response = await axios.default.post(
          `https://generativelanguage.googleapis.com/v1beta/models/${config.solutionModel || "gemini-2.0-flash"}:generateContent?key=${this.geminiApiKey}`,
          { contents: [{ role: "user", parts: [{ text: planPrompt }] }], generationConfig: { temperature: 0.2, maxOutputTokens: 4000 } }, { signal }
        );
        const responseData = response.data as GeminiResponse;
        planResponse = responseData.candidates?.[0]?.content?.parts?.[0]?.text || "";
      } else if (config.apiProvider === "anthropic") {
        if (!this.anthropicClient) return { success: false, error: "Anthropic Client not initialized." };
        const response = await this.anthropicClient.messages.create({
          model: config.solutionModel || "claude-3-7-sonnet-20250219",
          messages: [{ role: "user", content: planPrompt }],
          max_tokens: 4000,
          temperature: 0.2
        });
        if (Array.isArray(response.content) && response.content[0].type === 'text') {
            planResponse = response.content[0].text;
        }
      }

      if (signal.aborted) return { success: false, error: "Processing was canceled." };
      if (!planResponse) return { success: false, error: "Failed to generate a solution plan from the AI." };

      // == STAGE 2: THE "EXECUTION" ==
      const executionPrompt = `
You are an expert programmer executing a given plan. Your task is to write the final, optimal code and provide a concluding analysis based on the plan below.

**THE PLAN YOU MUST FOLLOW:**
---
${planResponse}
---

Your response MUST be structured with the following numbered sections, starting each on a new line with the exact headers:

1. Code:
\`\`\`${language}
// Provide the clean, optimal, and **heavily-commented** implementation in ${language}.
// The comments MUST narrate the implementation process, explaining the 'why' behind the code, the purpose of variables, and the logic of each step as if you are thinking aloud during an interview.
\`\`\`

2. Post-Solution Reflection (Thoughts):
* **Edge Cases:** List the critical edge cases the optimal solution handles or that should be tested.
* **Alternative Approaches:** Briefly mention one other valid approach and its trade-offs.

3. Time Complexity:
[Strictly start with O(notation) for the OPTIMAL solution, followed by a hyphen and a concise justification. Example: O(n) - The algorithm iterates through the input array a single time.]

4. Space Complexity:
[Strictly start with O(notation) for the OPTIMAL solution, followed by a hyphen and a concise justification. Example: O(n) - In the worst case, the hash map may store all unique elements from the input.]

---
**REMINDER OF PROBLEM LANGUAGE:** ${language}
---
`;

      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Writing optimal code with detailed explanations...",
          progress: 80
        });
      }

      let executionResponse = "";
      if (config.apiProvider === "openai") {
        if (!this.openaiClient) return { success: false, error: "OpenAI Client not initialized." };
        const response = await this.openaiClient.chat.completions.create({
          model: config.solutionModel || "gpt-4o",
          messages: [{ role: "system", content: "You are an expert programmer executing a plan." }, { role: "user", content: executionPrompt }],
          max_tokens: 4000,
          temperature: 0.2
        });
        executionResponse = response.choices[0].message.content || "";
      } else if (config.apiProvider === "gemini") {
        if (!this.geminiApiKey) return { success: false, error: "Gemini API Key not configured." };
        const response = await axios.default.post(
          `https://generativelanguage.googleapis.com/v1beta/models/${config.solutionModel || "gemini-2.0-flash"}:generateContent?key=${this.geminiApiKey}`,
          { contents: [{ role: "user", parts: [{ text: executionPrompt }] }], generationConfig: { temperature: 0.2, maxOutputTokens: 4000 } }, { signal }
        );
        const responseData = response.data as GeminiResponse;
        executionResponse = responseData.candidates?.[0]?.content?.parts?.[0]?.text || "";
      } else if (config.apiProvider === "anthropic") {
        if (!this.anthropicClient) return { success: false, error: "Anthropic Client not initialized." };
        const response = await this.anthropicClient.messages.create({
          model: config.solutionModel || "claude-3-7-sonnet-20250219",
          messages: [{ role: "user", content: executionPrompt }],
          max_tokens: 4000,
          temperature: 0.2
        });
        if (Array.isArray(response.content) && response.content[0].type === 'text') {
            executionResponse = response.content[0].text;
        }
      }

      // == COMBINE AND PARSE ==
      const sections = {
        walkthrough: planResponse.trim(), // The entire plan is the walkthrough
        code: `// Code in ${language} not found in AI response.`,
        reflection: "Reflection not found in AI response.",
        time_complexity: "Time complexity information was not provided.",
        space_complexity: "Space complexity information was not provided."
      };
      
      const executionContent = executionResponse.trim();

      const codeRegex = /1\.\s*Code:\s*```(?:[a-z0-9_.-]+)?\s*([\s\S]*?)\s*```/i;
      const reflectionRegex = /2\.\s*Post-Solution Reflection \(Thoughts\):([\s\S]*?)(?=\n3\.\s*Time Complexity:|$)/i;
      const timeComplexityRegex = /3\.\s*Time Complexity:\s*([\s\S]*?)(?=\n4\.\s*Space Complexity:|$)/i;
      const spaceComplexityRegex = /4\.\s*Space Complexity:\s*([\s\S]*?)(?=$)/i;

      const codeMatch = executionContent.match(codeRegex);
      if (codeMatch && codeMatch[1]) {
        sections.code = codeMatch[1].trim();
      }

      const reflectionMatch = executionContent.match(reflectionRegex);
      if (reflectionMatch && reflectionMatch[1]) {
        sections.reflection = reflectionMatch[1].trim();
      }

      const timeMatch = executionContent.match(timeComplexityRegex);
      if (timeMatch && timeMatch[1]) {
        sections.time_complexity = timeMatch[1].trim();
      }
      
      const spaceMatch = executionContent.match(spaceComplexityRegex);
      if (spaceMatch && spaceMatch[1]) {
        sections.space_complexity = spaceMatch[1].trim();
      }
      
      const formattedResponse = {
        walkthrough: sections.walkthrough,
        code: sections.code,
        reflection: sections.reflection,
        time_complexity: sections.time_complexity,
        space_complexity: sections.space_complexity
      };

      return { success: true, data: formattedResponse };
    } catch (error: any) {
      if (axios.isCancel(error)) {
        return {
          success: false,
          error: "Processing was canceled by the user."
        };
      }
      
      if (error?.response?.status === 401) {
        return {
          success: false,
          error: "Invalid API key. Please check your settings."
        };
      } else if (error?.response?.status === 429) {
        return {
          success: false,
          error: "API rate limit exceeded or insufficient credits. Please try again later."
        };
      }
      
      console.error("Solution generation error:", error);
      return { success: false, error: error.message || "Failed to generate solution" };
    }
  }

  private async processExtraScreenshotsHelper(
    screenshots: Array<{ path: string; data: string }>,
    signal: AbortSignal
  ) {
    try {
      const problemInfo = this.deps.getProblemInfo();
      const language = await this.getLanguage();
      const config = configHelper.loadConfig();
      const mainWindow = this.deps.getMainWindow();

      if (!problemInfo) {
        throw new Error("No problem info available");
      }

      // Update progress status
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Processing debug screenshots...",
          progress: 30
        });
      }

      // Prepare the images for the API call
      const imageDataList = screenshots.map(screenshot => screenshot.data);
      
      let debugContent = ""; // Initialize
      
      if (config.apiProvider === "openai") {
        if (!this.openaiClient) {
          return {
            success: false,
            error: "OpenAI API key not configured. Please check your settings."
          };
        }
        
        const messages = [
          {
            role: "system" as const, 
            content: `You are a coding interview assistant helping debug and improve solutions. Analyze these screenshots which include either error messages, incorrect outputs, or test cases, and provide detailed debugging help.

Your response MUST follow this exact structure with these section headers (use ### for headers):
### Issues Identified
- List each issue as a bullet point with clear explanation

### Specific Improvements and Corrections
- List specific code changes needed as bullet points

### Optimizations
- List any performance optimizations if applicable

### Explanation of Changes Needed
Here provide a clear explanation of why the changes are needed

### Key Points
- Summary bullet points of the most important takeaways

If you include code examples, use proper markdown code blocks with language specification (e.g. \`\`\`java).`
          },
          {
            role: "user" as const,
            content: [
              {
                type: "text" as const, 
                text: `I'm solving this coding problem: "${problemInfo.problem_statement}" in ${language}. I need help with debugging or improving my solution. Here are screenshots of my code, the errors or test cases. Please provide a detailed analysis with:
1. What issues you found in my code
2. Specific improvements and corrections
3. Any optimizations that would make the solution better
4. A clear explanation of the changes needed` 
              },
              ...imageDataList.map(data => ({
                type: "image_url" as const,
                image_url: { url: `data:image/png;base64,${data}` }
              }))
            ]
          }
        ];

        if (mainWindow) {
          mainWindow.webContents.send("processing-status", {
            message: "Analyzing code and generating debug feedback...",
            progress: 60
          });
        }

        const debugResponse = await this.openaiClient.chat.completions.create({
          model: config.debuggingModel || "gpt-4o",
          messages: messages,
          max_tokens: 4000,
          temperature: 0.2
        });
        
        debugContent = debugResponse.choices[0].message.content || "";
      } else if (config.apiProvider === "gemini")  {
        if (!this.geminiApiKey) {
          return {
            success: false,
            error: "Gemini API key not configured. Please check your settings."
          };
        }
        
        try {
          const debugPrompt = `
You are a coding interview assistant helping debug and improve solutions. Analyze these screenshots which include either error messages, incorrect outputs, or test cases, and provide detailed debugging help.

I'm solving this coding problem: "${problemInfo.problem_statement}" in ${language}. I need help with debugging or improving my solution.

YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE WITH THESE SECTION HEADERS:
### Issues Identified
- List each issue as a bullet point with clear explanation

### Specific Improvements and Corrections
- List specific code changes needed as bullet points

### Optimizations
- List any performance optimizations if applicable

### Explanation of Changes Needed
Here provide a clear explanation of why the changes are needed

### Key Points
- Summary bullet points of the most important takeaways

If you include code examples, use proper markdown code blocks with language specification (e.g. \`\`\`java).
`;

          const geminiMessages = [
            {
              role: "user",
              parts: [
                { text: debugPrompt },
                ...imageDataList.map(data => ({
                  inlineData: {
                    mimeType: "image/png",
                    data: data
                  }
                }))
              ]
            }
          ];

          if (mainWindow) {
            mainWindow.webContents.send("processing-status", {
              message: "Analyzing code and generating debug feedback with Gemini...",
              progress: 60
            });
          }

          const response = await axios.default.post(
            `https://generativelanguage.googleapis.com/v1beta/models/${config.debuggingModel || "gemini-2.0-flash"}:generateContent?key=${this.geminiApiKey}`,
            {
              contents: geminiMessages,
              generationConfig: {
                temperature: 0.2,
                maxOutputTokens: 4000
              }
            },
            { signal }
          );

          const responseData = response.data as GeminiResponse;
          
          if (!responseData.candidates || responseData.candidates.length === 0 || !responseData.candidates[0].content.parts[0].text) {
            throw new Error("Empty or invalid response from Gemini API for debugging");
          }
          
          debugContent = responseData.candidates[0].content.parts[0].text;
        } catch (error) {
          console.error("Error using Gemini API for debugging:", error);
          return {
            success: false,
            error: "Failed to process debug request with Gemini API. Please check your API key or try again later."
          };
        }
      } else if (config.apiProvider === "anthropic") {
        if (!this.anthropicClient) {
          return {
            success: false,
            error: "Anthropic API key not configured. Please check your settings."
          };
        }
        
        try {
          const debugPrompt = `
You are a coding interview assistant helping debug and improve solutions. Analyze these screenshots which include either error messages, incorrect outputs, or test cases, and provide detailed debugging help.

I'm solving this coding problem: "${problemInfo.problem_statement}" in ${language}. I need help with debugging or improving my solution.

YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE WITH THESE SECTION HEADERS:
### Issues Identified
- List each issue as a bullet point with clear explanation

### Specific Improvements and Corrections
- List specific code changes needed as bullet points

### Optimizations
- List any performance optimizations if applicable

### Explanation of Changes Needed
Here provide a clear explanation of why the changes are needed

### Key Points
- Summary bullet points of the most important takeaways

If you include code examples, use proper markdown code blocks with language specification.
`;

          const messages: Anthropic.Messages.MessageParam[] = [ 
            {
              role: "user" as const, 
              content: [
                {
                  type: "text" as const, 
                  text: debugPrompt
                },
                ...imageDataList.map(data => ({
                  type: "image" as const, 
                  source: {
                    type: "base64" as const,
                    media_type: "image/png" as const, 
                    data: data
                  }
                }))
              ]
            }
          ];

          if (mainWindow) {
            mainWindow.webContents.send("processing-status", {
              message: "Analyzing code and generating debug feedback with Claude...",
              progress: 60
            });
          }

          const response = await this.anthropicClient.messages.create({
            model: config.debuggingModel || "claude-3-7-sonnet-20250219", 
            max_tokens: 4000,
            messages: messages,
            temperature: 0.2
          });
          
          if (Array.isArray(response.content) && response.content.length > 0 && response.content[0].type === 'text') {
            debugContent = response.content[0].text;
          } else {
            throw new Error("Invalid response format from Anthropic API for debugging");
          }

        } catch (error: any) {
          console.error("Error using Anthropic API for debugging:", error);
          
          if (error.status === 429) {
            return {
              success: false,
              error: "Claude API rate limit exceeded. Please wait a few minutes before trying again."
            };
          } else if (error.status === 413 || (error.message && error.message.includes("token"))) {
            return {
              success: false,
              error: "Your screenshots contain too much information for Claude to process. Switch to OpenAI or Gemini in settings which can handle larger inputs."
            };
          }
          
          return {
            success: false,
            error: "Failed to process debug request with Anthropic API. Please check your API key or try again later."
          };
        }
      }
      
      
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Debug analysis complete",
          progress: 100
        });
      }

      let extractedCode = "// Debug mode - see analysis below";
      const codeMatch = debugContent.match(/```(?:[a-zA-Z]+)?([\s\S]*?)```/);
      if (codeMatch && codeMatch[1]) {
        extractedCode = codeMatch[1].trim();
      }

      let formattedDebugContent = debugContent;
      
      if (!debugContent.includes('# ') && !debugContent.includes('## ')) {
        formattedDebugContent = debugContent
          .replace(/issues identified|problems found|bugs found/i, '## Issues Identified')
          .replace(/code improvements|improvements|suggested changes/i, '## Code Improvements')
          .replace(/optimizations|performance improvements/i, '## Optimizations')
          .replace(/explanation|detailed analysis/i, '## Explanation');
      }

      const bulletPoints = formattedDebugContent.match(/(?:^|\n)[ ]*(?:[-*•]|\d+\.)[ ]+([^\n]+)/g);
      const thoughts = bulletPoints 
        ? bulletPoints.map(point => point.replace(/^[ ]*(?:[-*•]|\d+\.)[ ]+/, '').trim()).slice(0, 5)
        : ["Debug analysis based on your screenshots"];
      
      const response = {
        code: extractedCode,
        debug_analysis: formattedDebugContent,
        thoughts: thoughts, 
        time_complexity: "N/A - Debug mode",
        space_complexity: "N/A - Debug mode"
      };

      return { success: true, data: response };
    } catch (error: any) {
      console.error("Debug processing error:", error);
      return { success: false, error: error.message || "Failed to process debug request" };
    }
  }

  // --- START: NEW PROMPTS AND REFACTORED LOGIC ---

  public async processSingleGeneralProblem(newScreenshotPath: string, contextScreenshotPaths: string[]): Promise<{ success: boolean; data?: any; error?: string }> {
    this.currentProcessingAbortController = new AbortController();
    const { signal } = this.currentProcessingAbortController;

    try {
        const isInitialAnalysis = contextScreenshotPaths.length === 0;

        const newScreenshotData = fs.readFileSync(newScreenshotPath).toString("base64");
        const contextScreenshotsData = contextScreenshotPaths.map(p => fs.readFileSync(p).toString("base64"));
        const allImageData = [...contextScreenshotsData, newScreenshotData];

        // --- NEW PROMPTS ---
        const initialAnalysisPrompt = `**Your Persona:** You are "Oracle," a legendary, semi-mythical strategy partner. Your thinking is 10x deeper than a typical consultant. You are brutally concise. You generate structured, MECE, and deeply insightful output in bullet points. You must generate a high volume of quality ideas.

**Task:** The user has provided the first screenshot of a new case. Deconstruct it from first principles in an interview-ready flow.

**Output Format (Strictly Enforced):**

### 🎤 **Clarifying Questions & Case Framing**
* **Question 1 (Scope):** "Could you clarify the ultimate objective? Are we solving for profitability, market share, or something else? And over what time horizon?"
* **Question 2 (Context):** "What has the client already tried to do to address this issue, and what were the specific outcomes?"
* **Question 3 (Constraints):** "Are there any significant constraints we should be aware of, such as limitations on investment, brand identity, or existing partnerships?"
* **Problem Reframe:** Based on the prompt, I'll reframe the problem: "While this appears to be a [e.g., 'revenue growth'] challenge, it's more likely a [e.g., 'customer retention and value extraction'] problem at its core."

### 🌐 **Industry Snapshot: Outside-In Perspective**
* **Relevant Analog:** "This situation is reminiscent of the [e.g., 'US airline industry in the late 2000s'], where legacy carriers battled low-cost entrants. The key learning there was that competing on cost alone was a losing game; differentiation through customer experience was paramount."
* **Killer Stat:** "It's worth noting that in the [Client's Industry], a 5% improvement in customer retention has been shown to increase profitability by 25% to 95%. This suggests our focus should be on the existing customer base."
* **Niche Trend:** "A niche trend in this space is the rise of [e.g., 'hyper-personalization using AI']. While not mainstream, it's a disruptive force we must consider as a potential long-term play."

### 🗺️ **The Strategic Matrix: A Custom Framework**
*This is not a generic framework. Create a unique, MECE structure with as many vectors as needed for this case. For each sub-point, you MUST provide both a key question and a corresponding *Insight* or *Rationale* that explains why the question is important or what a potential answer might reveal.*

* **[Vector Name, e.g., "Profitability Levers"]**
    * **[Sub-Vector Name, e.g., "Pricing & Mix"]**
        * *Question:* Are we pricing based on value or cost?
        * *Insight:* Premium products often have inelastic demand; we may have untapped pricing power. A shift in product mix towards higher-margin SKUs could have a significant impact.
    * **[Sub-Vector Name, e.g., "Cost Optimization"]**
        * *Question:* Where are the non-obvious cost drivers in our value chain?
        * *Insight:* Often, complexity in processes (e.g., number of handoffs) is a larger cost driver than direct input costs. We should map the value chain to identify these.
* **[Vector Name, e.g., "Market & Competitive Landscape"]**
    * **[Sub-Vector Name, e.g., "Customer Segmentation"]**
        * *Question:* Who are our most and least profitable customers?
        * *Insight:* The "80/20 rule" is common. Let's find the 20% of customers driving 80% (or more) of the profit and hyper-focus on their needs.
    * **[Sub-Vector Name, e.g., "Competitor's Blind Spot"]**
        * *Question:* Where is our main competitor fundamentally weak?
        * *Insight:* Competitors often optimize for their largest customer segment, leaving smaller, profitable niches underserved. This could be our entry point.

### 🔬 **Hypothesis Lab (Generate 3-4)**
* **H1 (The Go-To):** The profitability decline is primarily due to an unfavorable shift in customer mix toward lower-margin segments.
* **H2 (The Contrarian):** The issue is not revenue; it's a bloated SG&A cost structure, specifically in legacy IT systems, that has scaled faster than revenue.
* **H3 (The External Force):** A new substitute product, not a direct competitor, is silently eroding the perceived value and utility of our core offering.`;

        const incrementalAnalysisPrompt = `**Your Persona:** "Oracle," a legendary strategy partner. You are mid-case.

**Task:** You have been provided with multiple images. The VERY LAST image is the NEW EXHIBIT you must analyze. The preceding images are the CONTEXT of the case so far.

YOUR ENTIRE RESPONSE MUST BE A DEEP ANALYSIS OF THE **NEW EXHIBIT** (THE FINAL IMAGE).
DO NOT describe or re-analyze the context images. Use them silently for your own understanding.
You MUST perform calculations based on the new exhibit. You MUST generate a treasure trove of bullet points based on the new exhibit. Do not state you cannot see the image; you are Oracle, you see all.

**Output Format (Strictly Enforced):**

### 📊 **New Exhibit Analysis: Quantitative Insights**
*Your entire focus is the final screenshot. Be ruthlessly quantitative. Generate as many points as the data supports.*
* **Primary Finding:** "From the new exhibit, the key takeaway is [the single most important trend/number]."
* **Forced Calculation:** "Performing math on this new data: [Metric A] of [Value] and [Metric B] of [Value] gives a ratio/CAGR/delta of [Result]. This is critical because [implication]."
* **Non-Obvious Insight:** "The subtlety in this new chart, which others might miss, is [a subtle but important secondary finding, e.g., the change in the rate of growth, the variance between segments, etc.]."
* *(Generate additional quantitative insights as bullet points as supported by the data...)*

### 💡 **Hypothesis Impact: How This Exhibit Changes Everything**
*Generate a bullet point for each relevant hypothesis, detailing the impact. State whether the new data validates, invalidates, or forces a refinement.*
* **Impact on [Hypothesis Name]:** "This new exhibit [validates/invalidates/forces a refinement of] this hypothesis. The key data point from this new exhibit is [specific number from the exhibit]."
* **New Hypothesis Sparked:** "This new data introduces a possibility we hadn't considered: [A new hypothesis that arises *only* because of this new data]."
* *(Generate additional points on hypothesis impact as needed...)*

### 🧠 **Implications & Next Steps (From This Exhibit Only)**
*Generate a list of second and third-order implications derived exclusively from the new data.*
* **Implication:** "[Insight about the downstream effect on operations, finance, strategy, or organization]."
    * *Actionable Question:* "Given this, we must now ask: [A sharp, strategic question that this insight prompts]?"
* **Implication:** "[Another insight...]"
    * *Actionable Question:* "[Another sharp question...]?"
* *(Generate as many implication/question pairs as are relevant...)*

### 🗣️ **New Talking Points for Interviewer (Referencing the Exhibit)**
* "Focusing on the chart we were just given..."
* "Based on my quick math from this new table..."
* "This exhibit is the most critical piece of information yet, because it tells us..."

**Final Check: Did you focus your entire written analysis on the LAST image provided? Yes.**
`;

        const prompt = isInitialAnalysis ? initialAnalysisPrompt : incrementalAnalysisPrompt;
        const userMessage = isInitialAnalysis
            ? "Analyze the case presented in this image."
            : "Analyze the new information in the final image, using the previous images for context.";

        const config = configHelper.loadConfig();
        let responseContent;
        const model = config.solutionModel || "gpt-4o";

        if (config.apiProvider === 'openai' && this.openaiClient) {
            const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
                { role: "system", content: prompt },
                {
                    role: "user",
                    content: [
                        { type: "text", text: userMessage },
                        ...allImageData.map(data => ({ type: "image_url" as const, image_url: { url: `data:image/png;base64,${data}` } }))
                    ]
                }
            ];
            const response = await this.openaiClient.chat.completions.create({ model, messages, max_tokens: 4000, temperature: 0.4 }, { signal });
            responseContent = response.choices[0].message.content;
        } else {
            return { success: false, error: "This AI provider is not configured for this action." };
        }

        return { success: true, data: { solution: responseContent } };

    } catch (error: any) {
        console.error("Error in processSingleGeneralProblem:", error);
        if (axios.isCancel(error)) return { success: false, error: "Processing was canceled." };
        return { success: false, error: error.message || "Failed to process general problem." };
    } finally {
        this.currentProcessingAbortController = null;
    }
  }
  
  public async summarizeAnalyses(analyses: string[]): Promise<{ success: boolean; data?: any; error?: string }> {
    const summarizationPrompt = `**Your Persona:** "Oracle," a legendary strategy partner. The analysis is complete.

**Task:** Synthesize all previous analyses into a powerful, data-driven, and "boardroom-ready" final recommendation. Structure it as a compelling narrative for a CEO, using only bullet points for easy tracking.

**Output Format (Strictly Enforced):**

### 🔑 **The Executive Synthesis: The One Big Idea**
* **The Narrative:**
    * Our analysis reveals that the core issue is not [the assumed problem, e.g., 'a weak market'], but a fundamental misalignment between the company's sales strategy and its profitability goals.
    * We initially believed the primary threat was [competitor X], but the data clearly shows the threat is internal: a "growth at all costs" mindset.
    * The critical turning point was revealed in Exhibit [Number], which showed that our fastest-growing segment is also our least profitable, with a negative customer lifetime value.
* **The Core Recommendation:**
    * We must pivot immediately from a strategy of 'unprofitable growth' to one of 'profitable growth'.
    * This involves a surgical focus on our high-value customer segments and a disciplined approach to cost and investment.

### 📈 **The "Thrust & Vector" Action Plan**
*This plan must be data-driven. Generate as many strategic "Thrusts" as are necessary to address the core problem, each with a clear action, rationale, and KPI.*
* **Thrust: [Name of Initiative, e.g., "Stabilize the Core"] (Timeline: e.g., 0-3 Months)**
    * **Action:** [Specific action, e.g., "Restructure pricing for Segment A"].
    * **Rationale:** "This addresses the [Y]% margin erosion we calculated from Exhibit 2."
    * **KPI:** Measure success by a return to [Z]% gross margin.
* **Thrust: [Name of Initiative, e.g., "Ignite New Growth"] (Timeline: e.g., 3-9 Months)**
    * **Action:** [Specific action, e.g., "Launch a pilot program for Product B in the Asia-Pacific market"].
    * **Rationale:** "This capitalizes on the [W]% market growth and higher WTP we identified in Exhibit 4."
    * **KPI:** Target [V] units sold and a customer acquisition cost below $[U].
* *(Generate additional Thrusts as needed...)*

### 🎲 **Pre-Mortem: What Could Go Wrong**
*Generate as many strategic "Risks" as are necessary to forsee any problems in our analysis and for the case in this format below:*
* **Risk:** "[e.g., 'Cultural resistance from the sales team','Competitor C responds with a price war',etc]."
    * **Mitigation:** "[e.g., 'Redesign sales incentives in Q1','Prepare a 'war chest' and a counter-messaging campaign',etc]."
* *(Generate additional risks and mitigations as deemed important for this case and our insights...)*

### 🔭 **Beyond the Horizon: The Untapped Opportunity**
* **Final Insight:** The capabilities developed during this turnaround—particularly our new expertise in customer segmentation and value-based pricing—will unlock a significant opportunity.
* **The Prize:** We can leverage this to enter the adjacent [e.g., 'B2B enterprise software'] market, which is currently a potential $[X]B prize and a natural fit for our newly honed skills.
`;

    const userMessage = `Here are the sequential analyses of the case. Please synthesize them into a final recommendation:\n\n---\n\n` + analyses.join('\n\n---\n\n');
    
    this.currentProcessingAbortController = new AbortController();
    const { signal } = this.currentProcessingAbortController;

    try {
        const config = configHelper.loadConfig();
        let summaryContent;
        const model = config.solutionModel || "gpt-4o";

        if (config.apiProvider === 'openai' && this.openaiClient) {
             const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
                { role: "system", content: summarizationPrompt },
                { role: "user", content: userMessage }
            ];
            const response = await this.openaiClient.chat.completions.create({ model, messages, max_tokens: 4000, temperature: 0.4 }, { signal });
            summaryContent = response.choices[0].message.content;
        } else {
             return { success: false, error: "This AI provider is not configured for summarization." };
        }

        return { success: true, data: { summary: summaryContent } };

    } catch (error: any) {
        console.error("Error in summarizeAnalyses:", error);
        if (axios.isCancel(error)) return { success: false, error: "Summarization was canceled." };
        return { success: false, error: "Failed to generate summary." };
    } finally {
      this.currentProcessingAbortController = null;
    }
  }

  // --- END: NEW PROMPTS AND REFACTORED LOGIC ---

  public cancelOngoingRequests(): void {
    let wasCancelled = false

    if (this.currentProcessingAbortController) {
      this.currentProcessingAbortController.abort()
      this.currentProcessingAbortController = null
      wasCancelled = true
    }

    if (this.currentExtraProcessingAbortController) {
      this.currentExtraProcessingAbortController.abort()
      this.currentExtraProcessingAbortController = null
      wasCancelled = true
    }

    this.deps.setHasDebugged(false)
    this.deps.setProblemInfo(null)

    const mainWindow = this.deps.getMainWindow()
    if (wasCancelled && mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS)
    }
  }
}
