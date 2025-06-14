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

  // --- START: MODIFIED SECTION FOR NEW LOGIC ---

  public async processSingleGeneralProblem(newScreenshotPath: string, contextScreenshotPaths: string[]): Promise<{ success: boolean; data?: any; error?: string }> {
    this.currentProcessingAbortController = new AbortController();
    const { signal } = this.currentProcessingAbortController;

    try {
      const isInitialAnalysis = contextScreenshotPaths.length === 0;

      const newScreenshotData = fs.readFileSync(newScreenshotPath).toString("base64");
      const contextScreenshotsData = contextScreenshotPaths.map(p => fs.readFileSync(p).toString("base64"));
      const allImageData = [...contextScreenshotsData, newScreenshotData];

      const initialAnalysisPrompt = `You are a BCG case interview strategist. This is the first piece of information for a new case, likely the problem statement. Generate a DENSE stream of differentiated talking points based *only* on this initial view. Every line should make an interviewer think "I haven't heard this angle before."

**GENERATE THIS EXACT FORMAT:**

🎯 **OPENING SALVO** (30 seconds of gold)
- Reframe: "This looks like [obvious thing] but it's actually [sophisticated insight]"
- Hidden assumption to challenge: "[What client believes] but have we validated [contrarian view]?"
- Industry parallel: "Similar to how [Company X] faced [analogous situation] in [Year]"
- Strategic altitude: "Before diving into [tactical issue], the real question is [strategic issue]"

📐 **FRAMEWORK 2.0** (A custom framework for THIS case)
**[Custom Framework Name for THIS case]**
1. **[Bucket 1]**:
    - Key questions: [2-3 specific, non-obvious questions]
    - Data needed: [Specific metrics others won't ask for]
2. **[Bucket 2]**:
    - Key questions: [2-3 specific questions]
    - Watch for: [Common blind spot]
3. **[Bucket 3]**:
    - Key questions: [2-3 specific questions]
    - Risk factor: [What could break]

💡 **HYPOTHESIS BATTERY** (Pick 2-3 to test)
- H1: [Specific, testable hypothesis with clear data needs]
- H2: [Counter-hypothesis that flips conventional wisdom]
- H3: [Systems-level hypothesis about root cause]

⚡ **CRITICAL QUESTIONS TO ASK NOW**
1. "What would have to be true for [contrarian strategy] to work?"
2. "Where is the client's mental model misaligned with market reality?"
3. "What's the one metric that, if it moved 20%, would change everything?"`;

      const incrementalAnalysisPrompt = `You are a BCG case interview strategist continuing a case analysis. You have the context of the previous information (in the first image(s)). Now, a new piece of data has been provided (in the final image).

Your task is to **analyze the new information** and synthesize it with the existing context. FOCUS ON WHAT IS NEW. What does this new chart/table/data tell you? How does it confirm, deny, or modify your previous hypotheses?

**GENERATE THIS EXACT FORMAT:**

📊 **NEW DATA - ADVANCED READS**
- Pattern break: "The new data shows [X] but notice how [segment/time] breaks the previous pattern"
- Missing middle: "We now see [extremes], but what about [middle segment]?"
- Ratio insight: "[New Metric A] ÷ [Existing Metric B] reveals [hidden dynamic]"
- Trajectory trap: "The new trend suggests [X], but [leading indicator] from before warns of [Y]"
- Cross-reference: "Combining this new chart with the previous information exposes [synthesis]"

💡 **HYPOTHESIS UPDATE**
- **Confirmed/Modified H1:** [How the new data impacts H1]
- **New Hypothesis H4:** [A new hypothesis prompted by this data]
- **Data Conflict:** [Does this data conflict with anything seen before? How?]

🔍 **INTERVIEW JUJITSU** (New questions this data lets you ask)
- "This data on [X] is interesting, but it makes me question our assumption about [Y]. Can we double-check that?"
- "Given this, the bottleneck seems to have shifted from [old constraint] to [new constraint]. Is that correct?"

⚡ **CRITICAL QUESTIONS RAISED BY THIS DATA**
1. "This implies we need to [action]. What capability do we need to build to do that?"
2. "If this trend continues, what's the biggest risk in the next 6 months?"`;

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
    const summarizationPrompt = `You are a BCG case interview strategist who has just completed a multi-stage case analysis. You will be given the sequence of your own analyses.

Your task is to synthesize ALL the insights into a final, impactful recommendation for the client. The output must be structured, actionable, and persuasive.

**GENERATE THIS EXACT FORMAT:**

🚀 **CLOSING CRESCENDO** (Land the plane with impact)
- **Core Synthesis:** "The central issue isn't [obvious problem], but rather [deeper, non-obvious connection between all findings]."
- **Strategic Recommendation:** "Therefore, we recommend a three-pronged approach:
    1. **Do This First (Next 90 Days):** [Specific, immediate action] to capture [specific value].
    2. **Then Do This (Next 6-12 Months):** [Second, larger action] to build [specific capability].
    3. **And This for the Long Term:** [Third, strategic move] to secure [long-term advantage]."
- **Key Risk & Mitigation:** "The biggest risk to this plan is [specific risk, e.g., competitor reaction, customer adoption]. We can hedge this by [specific monitoring metric and threshold]."
- **The 'So What' for the Client:** "Successfully executing this unlocks the ability to [bigger, adjacent opportunity], transforming the business from [current state] to [future state]."`;

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

  // NOTE: The old processGeneralProblem and processGeneralProblemLLM have been removed.

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