/**
 * MemeGuard AI (MSCMGTB) — Centralized Project Configuration
 * Single source of truth for project metadata, architecture, and engine settings.
 */

export interface ProjectConfig {
  projectName: string;
  projectSubtitle: string;
  projectAcronym: string;
  applicationName: string;
  modelArchitecture: string;
  ocrEnabled: boolean;
  ocrEngine: string;
  calibratedThreshold: number;
}

export const projectConfig: ProjectConfig = {
  projectName: 'Multimodal Social Media Content Moderation Using Hybrid Graph Theory and Bio-Inspired Optimization',
  projectSubtitle: 'Multimodal Social Media Content Moderation',
  projectAcronym: 'MSCMGTB',
  applicationName: 'MemeGuard AI',
  modelArchitecture: 'ResNet50 + BERT + BAM',
  ocrEnabled: true,
  ocrEngine: 'EasyOCR (CRAFT + CRNN)',
  calibratedThreshold: 0.41,
};
